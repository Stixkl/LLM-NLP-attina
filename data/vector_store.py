"""Módulo de base de datos vectorial para Attina usando ChromaDB.

Responsabilidades:
- Indexar mensajes del dataset en ChromaDB con embeddings de texto.
- Exponer búsqueda semántica sobre las conversaciones.
- Integrar con el DataLoader existente.
"""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .schema import Message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

_CHROMA_DIR = Path(__file__).resolve().parent.parent / ".chromadb"
_COLLECTION_NAME = "attina_conversations"
_EMBED_MODEL = "all-MiniLM-L6-v2"   # modelo liviano local (sentence-transformers)


# ---------------------------------------------------------------------------
# Cliente y colección (singleton)
# ---------------------------------------------------------------------------

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_embedding_fn() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=_EMBED_MODEL
    )


def get_client() -> chromadb.PersistentClient:
    """Retorna el cliente ChromaDB (crea el directorio si no existe)."""
    global _client
    if _client is None:
        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(_CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB inicializado en: %s", _CHROMA_DIR)
    return _client


def get_collection() -> chromadb.Collection:
    """Retorna (o crea) la colección principal de conversaciones."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=_get_embedding_fn(),
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Colección '%s' lista. Documentos indexados: %d",
            _COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


# ---------------------------------------------------------------------------
# Indexación
# ---------------------------------------------------------------------------

def _message_to_document(msg: Message) -> tuple[str, str, dict]:
    """Convierte un Message en (id, texto_indexable, metadata)."""
    doc_id = hashlib.md5(msg.id.encode()).hexdigest()

    # Texto enriquecido para el embedding
    parts = [msg.text]
    if msg.author.username:
        parts.append(f"autor: {msg.author.username}")
    if msg.location and msg.location.country:
        parts.append(f"país: {msg.location.country}")
    if msg.tags:
        parts.append(f"etiquetas: {', '.join(msg.tags)}")
    if msg.keywords:
        parts.append(f"palabras clave: {', '.join(msg.keywords)}")
    text = " | ".join(parts)

    metadata = {
        "message_id": msg.id,
        "author_id": msg.author.id,
        "author_username": msg.author.username or "",
        "source": msg.source.value,
        "message_type": msg.message_type.value,
        "created_at": msg.created_at.isoformat(),
        "parent_id": msg.parent_id or "",
        "thread_id": msg.thread_id or "",
        "country": (msg.location.country if msg.location else "") or "",
        "language": msg.language or "",
        "sentiment": float(msg.sentiment) if msg.sentiment is not None else 0.0,
        "likes": msg.likes,
        "has_media": msg.has_media,
        "keywords": ", ".join(msg.keywords),
        "tags": ", ".join(msg.tags),
    }

    return doc_id, text, metadata


def index_messages(messages: list[Message], batch_size: int = 100) -> int:
    """
    Indexa una lista de mensajes en ChromaDB.

    Solo inserta documentos que aún no están en la colección (upsert por ID).

    Returns:
        Número total de documentos en la colección tras la indexación.
    """
    collection = get_collection()
    total = 0

    for i in range(0, len(messages), batch_size):
        batch = messages[i : i + batch_size]
        ids, documents, metadatas = [], [], []

        for msg in batch:
            if not msg.text or not msg.text.strip():
                continue
            doc_id, text, metadata = _message_to_document(msg)
            ids.append(doc_id)
            documents.append(text)
            metadatas.append(metadata)

        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            total += len(ids)
            logger.debug("Lote indexado: %d documentos (batch %d)", len(ids), i // batch_size + 1)

    count = collection.count()
    logger.info("Indexación completa. Total en colección: %d", count)
    return count


def index_raw_dicts(raw_messages: list[dict], batch_size: int = 100) -> int:
    """
    Indexa desde dicts crudos del JSON (sin pasar por Message).
    Útil para datasets parcialmente compatibles con el schema.
    """
    from .schema import Message as M
    messages = []
    for item in raw_messages:
        try:
            messages.append(M.from_dict(item))
        except Exception as exc:
            logger.warning("Omitido registro malformado (%s): %s", item.get("id", "?"), exc)
    return index_messages(messages, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Búsqueda semántica
# ---------------------------------------------------------------------------

def search(
    query: str,
    n_results: int = 5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Busca mensajes semánticamente similares a `query`.

    Args:
        query:     Texto libre para buscar.
        n_results: Número de resultados a devolver.
        where:     Filtro de metadata ChromaDB (ej: {"country": "Colombia"}).

    Returns:
        Lista de dicts con {message_id, text, metadata, distance}.
    """
    collection = get_collection()

    if collection.count() == 0:
        logger.warning("La colección está vacía. Indexa datos antes de buscar.")
        return []

    kwargs: dict = {"query_texts": [query], "n_results": min(n_results, collection.count())}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "message_id": results["metadatas"][0][i].get("message_id", ""),
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return output


def search_by_sentiment(
    query: str,
    sentiment_min: float = -1.0,
    sentiment_max: float = 1.0,
    n_results: int = 5,
) -> list[dict]:
    """Búsqueda semántica filtrada por rango de sentimiento."""
    return search(
        query=query,
        n_results=n_results,
        where={
            "$and": [
                {"sentiment": {"$gte": sentiment_min}},
                {"sentiment": {"$lte": sentiment_max}},
            ]
        },
    )


def search_by_country(query: str, country: str, n_results: int = 5) -> list[dict]:
    """Búsqueda semántica filtrada por país."""
    return search(query=query, n_results=n_results, where={"country": country})


# ---------------------------------------------------------------------------
# Utilidades de gestión
# ---------------------------------------------------------------------------

def get_stats() -> dict:
    """Retorna estadísticas de la colección ChromaDB."""
    collection = get_collection()
    count = collection.count()
    return {
        "collection": _COLLECTION_NAME,
        "total_documents": count,
        "embed_model": _EMBED_MODEL,
        "persist_dir": str(_CHROMA_DIR),
    }


def clear_collection() -> None:
    """Elimina y recrea la colección (borra todos los embeddings)."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(_COLLECTION_NAME)
        logger.info("Colección '%s' eliminada.", _COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    get_collection()  # Recrea vacía
