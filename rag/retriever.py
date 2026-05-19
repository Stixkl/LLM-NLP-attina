"""Módulo de recuperación RAG para Attina.

Recupera fragmentos relevantes de dos fuentes:
  1. Mensajes del dataset (ChromaDB — colección 'attina_conversations')
  2. Documentos externos (PDFs, artículos, reportes) — colección 'attina_documents'

La función principal `retrieve()` devuelve una lista unificada de fragmentos
ordenados por relevancia, lista para ser inyectada en el prompt de Gemini.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_CHROMA_DIR = Path(__file__).resolve().parent.parent / ".chromadb"
_CONVERSATIONS_COLLECTION = "attina_conversations"
_DOCUMENTS_COLLECTION = "attina_documents"
_EMBED_MODEL = "all-MiniLM-L6-v2"

# Cuántos fragmentos traer de cada fuente por defecto
_DEFAULT_K_CONVERSATIONS = 4
_DEFAULT_K_DOCUMENTS = 3


# ---------------------------------------------------------------------------
# Shared ChromaDB client
# ---------------------------------------------------------------------------

_client: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(_CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _embed_fn() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)


def _get_collection(name: str) -> chromadb.Collection:
    return _get_client().get_or_create_collection(
        name=name,
        embedding_function=_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Data model for retrieved fragments
# ---------------------------------------------------------------------------

@dataclass
class Fragment:
    """Fragmento de texto recuperado con su metadata y puntuación."""
    source_type: str          # "conversation" | "document"
    source_id: str            # message_id o doc_id
    source_label: str         # nombre legible: autor/fuente o título del doc
    text: str                 # texto del fragmento (limpio, sin metadata)
    score: float              # similitud coseno [0, 1]
    metadata: dict = field(default_factory=dict)

    def as_context_block(self) -> str:
        """Formato compacto para insertar en el prompt de Gemini."""
        tag = "CONVERSACIÓN" if self.source_type == "conversation" else "DOCUMENTO"
        score_pct = f"{self.score * 100:.0f}%"
        return f"[{tag} | {self.source_label} | relevancia {score_pct}]\n{self.text}"


# ---------------------------------------------------------------------------
# Retrieval from conversations
# ---------------------------------------------------------------------------

def _retrieve_conversations(query: str, k: int) -> list[Fragment]:
    col = _get_collection(_CONVERSATIONS_COLLECTION)
    if col.count() == 0:
        logger.warning("Colección de conversaciones vacía.")
        return []

    results = col.query(
        query_texts=[query],
        n_results=min(k, col.count()),
    )

    fragments = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        # El doc indexado incluye prefijos "autor: X | país: Y | ..."
        # Extraemos solo el texto original (la primera parte antes del primer " | ")
        raw_text = doc.split(" | ")[0] if " | " in doc else doc
        score = max(0.0, 1.0 - results["distances"][0][i])

        author = meta.get("author_username") or meta.get("author_id", "desconocido")
        country = meta.get("country", "")
        label = f"@{author}" + (f" · {country}" if country else "")

        fragments.append(Fragment(
            source_type="conversation",
            source_id=meta.get("message_id", ""),
            source_label=label,
            text=raw_text.strip(),
            score=score,
            metadata=meta,
        ))

    return fragments


# ---------------------------------------------------------------------------
# Retrieval from external documents
# ---------------------------------------------------------------------------

def _retrieve_documents(query: str, k: int) -> list[Fragment]:
    col = _get_collection(_DOCUMENTS_COLLECTION)
    if col.count() == 0:
        logger.debug("Colección de documentos externos vacía.")
        return []

    results = col.query(
        query_texts=[query],
        n_results=min(k, col.count()),
    )

    fragments = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        score = max(0.0, 1.0 - results["distances"][0][i])
        label = meta.get("title") or meta.get("filename", "documento")
        page = meta.get("page")
        if page:
            label += f" (p.{page})"

        fragments.append(Fragment(
            source_type="document",
            source_id=meta.get("doc_id", ""),
            source_label=label,
            text=doc.strip(),
            score=score,
            metadata=meta,
        ))

    return fragments


# ---------------------------------------------------------------------------
# Unified retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k_conversations: int = _DEFAULT_K_CONVERSATIONS,
    k_documents: int = _DEFAULT_K_DOCUMENTS,
    min_score: float = 0.25,
) -> list[Fragment]:
    """
    Recupera los fragmentos más relevantes para `query` de ambas fuentes.

    Args:
        query:           Pregunta o texto del usuario.
        k_conversations: Máx. fragmentos de conversaciones a recuperar.
        k_documents:     Máx. fragmentos de documentos externos a recuperar.
        min_score:       Umbral mínimo de similitud (0-1). Fragmentos por debajo
                         se descartan para no añadir ruido al contexto.

    Returns:
        Lista de Fragment ordenada por score descendente.
    """
    conv_fragments = _retrieve_conversations(query, k=k_conversations)
    doc_fragments = _retrieve_documents(query, k=k_documents)

    all_fragments = conv_fragments + doc_fragments

    # Filtrar por umbral de relevancia mínima
    filtered = [f for f in all_fragments if f.score >= min_score]

    if not filtered:
        logger.info("Ningún fragmento superó el umbral min_score=%.2f. Usando top-3 sin filtro.", min_score)
        filtered = sorted(all_fragments, key=lambda f: f.score, reverse=True)[:3]

    # Ordenar globalmente por score
    filtered.sort(key=lambda f: f.score, reverse=True)

    logger.info(
        "RAG retrieve: query=%r → %d conv + %d docs → %d útiles",
        query[:60],
        len(conv_fragments),
        len(doc_fragments),
        len(filtered),
    )

    return filtered


def build_context_string(fragments: list[Fragment], max_chars: int = 6000) -> str:
    """
    Construye el bloque de contexto para inyectar en el prompt.

    Trunca si el conjunto de fragmentos supera `max_chars` para no
    desbordar la ventana de contexto de Gemini.
    """
    if not fragments:
        return "(Sin contexto recuperado)"

    blocks = []
    total = 0
    for frag in fragments:
        block = frag.as_context_block()
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    return "\n\n---\n\n".join(blocks)
