"""MCP Service: Búsqueda Semántica sobre ChromaDB.

Endpoint: POST /analisis/semantico

Permite al agente buscar mensajes por similitud semántica,
con filtros opcionales de sentimiento y país.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.vector_store import search, search_by_country, search_by_sentiment

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas de entrada / salida
# ---------------------------------------------------------------------------

class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Texto libre para buscar semánticamente")
    n_results: int = Field(5, ge=1, le=20, description="Número de resultados")
    country: Optional[str] = Field(None, description="Filtrar por país")
    sentiment_min: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Sentimiento mínimo")
    sentiment_max: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Sentimiento máximo")


class SemanticResult(BaseModel):
    message_id: str
    text: str
    author: str
    country: str
    language: str
    sentiment: float
    source: str
    created_at: str
    similarity_score: float   # 1 - distance (coseno)


class SemanticSearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SemanticResult]
    filters_applied: dict


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/analisis/semantico", response_model=SemanticSearchResponse)
async def analisis_semantico(request: SemanticSearchRequest) -> SemanticSearchResponse:
    """
    Búsqueda semántica sobre las conversaciones indexadas en ChromaDB.

    Encuentra mensajes conceptualmente similares a la consulta,
    ordenados por relevancia (similitud coseno).
    """
    try:
        # Elegir función de búsqueda según filtros
        if request.country:
            raw_results = search_by_country(
                query=request.query,
                country=request.country,
                n_results=request.n_results,
            )
        elif request.sentiment_min is not None or request.sentiment_max is not None:
            raw_results = search_by_sentiment(
                query=request.query,
                sentiment_min=request.sentiment_min if request.sentiment_min is not None else -1.0,
                sentiment_max=request.sentiment_max if request.sentiment_max is not None else 1.0,
                n_results=request.n_results,
            )
        else:
            raw_results = search(query=request.query, n_results=request.n_results)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda vectorial: {exc}") from exc

    results = [
        SemanticResult(
            message_id=r["message_id"],
            text=r["text"],
            author=r["metadata"].get("author_username", ""),
            country=r["metadata"].get("country", ""),
            language=r["metadata"].get("language", ""),
            sentiment=r["metadata"].get("sentiment", 0.0),
            source=r["metadata"].get("source", ""),
            created_at=r["metadata"].get("created_at", ""),
            similarity_score=round(1.0 - r["distance"], 4),
        )
        for r in raw_results
    ]

    filters_applied = {}
    if request.country:
        filters_applied["country"] = request.country
    if request.sentiment_min is not None:
        filters_applied["sentiment_min"] = request.sentiment_min
    if request.sentiment_max is not None:
        filters_applied["sentiment_max"] = request.sentiment_max

    return SemanticSearchResponse(
        query=request.query,
        total_results=len(results),
        results=results,
        filters_applied=filters_applied,
    )
