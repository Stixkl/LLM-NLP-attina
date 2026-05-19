"""Ingestión de documentos externos en ChromaDB.

Soporta: PDF, TXT, MD
Los documentos se dividen en chunks con solapamiento y se indexan
en la colección 'attina_documents', separada de las conversaciones.

Uso:
    python -m rag.ingestor --file docs/reporte.pdf
    python -m rag.ingestor --dir docs/externos/
    python -m rag.ingestor --stats
    python -m rag.ingestor --clear
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Asegurar imports desde raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import _DOCUMENTS_COLLECTION, _embed_fn, _get_client

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 500       # caracteres por chunk
_CHUNK_OVERLAP = 100    # solapamiento entre chunks consecutivos


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Divide texto en chunks con solapamiento. Respeta límites de párrafo cuando es posible."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Intentar cortar en límite de párrafo o frase
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                pos = text.rfind(sep, start, end)
                if pos != -1 and pos > start + overlap:
                    end = pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# Text extraction per format
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> str:
    """Extrae texto de un PDF usando pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Instala pypdf: pip install pypdf --break-system-packages")

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_text(path: Path) -> str:
    """Extrae texto de TXT o MD."""
    return path.read_text(encoding="utf-8", errors="replace")


def extract_text(path: Path) -> str:
    """Dispatcher por extensión."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix in {".txt", ".md"}:
        return _extract_text(path)
    else:
        raise ValueError(f"Formato no soportado: {suffix}. Usa PDF, TXT o MD.")


# ---------------------------------------------------------------------------
# Indexación
# ---------------------------------------------------------------------------

def _doc_id(path: Path, chunk_index: int) -> str:
    key = f"{path.name}:{chunk_index}"
    return hashlib.md5(key.encode()).hexdigest()


def ingest_file(path: Path, title: str | None = None) -> int:
    """
    Indexa un archivo en la colección de documentos externos.

    Returns:
        Número de chunks indexados.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info("Procesando: %s", path.name)
    raw_text = extract_text(path)
    chunks = _chunk_text(raw_text)

    if not chunks:
        logger.warning("Archivo vacío o sin texto extraíble: %s", path.name)
        return 0

    doc_title = title or path.stem.replace("_", " ").replace("-", " ").title()

    collection = _get_client().get_or_create_collection(
        name=_DOCUMENTS_COLLECTION,
        embedding_function=_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(_doc_id(path, i))
        documents.append(chunk)
        metadatas.append({
            "doc_id": _doc_id(path, 0),       # ID de documento padre
            "filename": path.name,
            "title": doc_title,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "file_type": path.suffix.lstrip("."),
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("✅ %s → %d chunks indexados", path.name, len(chunks))
    return len(chunks)


def ingest_directory(directory: Path, recursive: bool = False) -> dict[str, int]:
    """
    Indexa todos los archivos soportados de un directorio.

    Returns:
        Dict {nombre_archivo: chunks_indexados}
    """
    directory = Path(directory)
    pattern = "**/*" if recursive else "*"
    supported = {".pdf", ".txt", ".md"}

    results = {}
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in supported:
            try:
                n = ingest_file(path)
                results[path.name] = n
            except Exception as exc:
                logger.error("Error procesando %s: %s", path.name, exc)
                results[path.name] = 0

    return results


def get_docs_stats() -> dict:
    """Estadísticas de la colección de documentos."""
    col = _get_client().get_or_create_collection(
        name=_DOCUMENTS_COLLECTION,
        embedding_function=_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )
    count = col.count()
    # Contar documentos únicos (no chunks)
    if count > 0:
        all_meta = col.get(include=["metadatas"])["metadatas"]
        unique_files = len({m.get("filename", "") for m in all_meta})
    else:
        unique_files = 0

    return {
        "collection": _DOCUMENTS_COLLECTION,
        "total_chunks": count,
        "unique_files": unique_files,
    }


def clear_docs() -> None:
    """Elimina y recrea la colección de documentos externos."""
    try:
        _get_client().delete_collection(_DOCUMENTS_COLLECTION)
        logger.info("Colección '%s' eliminada.", _DOCUMENTS_COLLECTION)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingestor de documentos externos → ChromaDB")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Ruta a un archivo PDF, TXT o MD")
    group.add_argument("--dir", help="Directorio con documentos a indexar")
    group.add_argument("--stats", action="store_true", help="Mostrar estadísticas")
    group.add_argument("--clear", action="store_true", help="Eliminar colección de documentos")

    parser.add_argument("--title", help="Título del documento (solo con --file)")
    parser.add_argument("--recursive", action="store_true", help="Buscar recursivamente en subdirectorios")

    args = parser.parse_args()

    if args.stats:
        stats = get_docs_stats()
        print("\n=== Documentos externos en ChromaDB ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print()

    elif args.clear:
        clear_docs()
        print("Colección de documentos eliminada.")

    elif args.file:
        n = ingest_file(Path(args.file), title=args.title)
        print(f"Indexados {n} chunks de '{args.file}'")

    elif args.dir:
        results = ingest_directory(Path(args.dir), recursive=args.recursive)
        total = sum(results.values())
        print(f"\nIndexados {total} chunks de {len(results)} archivos:")
        for fname, n in results.items():
            print(f"  {fname}: {n} chunks")


if __name__ == "__main__":
    main()
