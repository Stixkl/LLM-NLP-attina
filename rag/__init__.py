"""Módulo RAG de Attina: recuperación y generación aumentada."""

from .retriever import Fragment, build_context_string, retrieve
from .ingestor import ingest_file, ingest_directory, get_docs_stats

__all__ = [
    "Fragment",
    "retrieve",
    "build_context_string",
    "ingest_file",
    "ingest_directory",
    "get_docs_stats",
]
