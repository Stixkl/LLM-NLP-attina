"""LangChain Tools que invocan los MCP Services de Attina."""

import httpx
from langchain_core.tools import tool
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services import config

MCP_BASE_URL = f"http://{config.MCP_HOST}:{config.MCP_PORT}"
TIMEOUT = 30.0


def _post(endpoint: str, payload: dict) -> dict:
    """Realiza un POST al MCP service y retorna el JSON de respuesta."""
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(f"{MCP_BASE_URL}{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()


@tool
def tool_resumen(tematica: Optional[str] = None) -> dict:
    """
    Genera un resumen ejecutivo de las conversaciones del dataset.
    Identifica la temática principal, posturas (positiva/negativa/neutral)
    y palabras clave. Úsala cuando el usuario pida un resumen, síntesis
    o visión general de la conversación.

    Args:
        tematica: Tema específico sobre el que enfocar el resumen (opcional).

    Returns:
        Diccionario con resumen, temática principal, posturas y palabras clave.
    """
    payload: dict = {}
    if tematica:
        payload["tematica"] = tematica
    return _post("/analisis/resumen", payload)


@tool
def tool_geografico() -> dict:
    """
    Analiza la distribución geográfica de las conversaciones.
    Agrupa por zonas cardinales (Norte/Sur/Este/Oeste/Centro),
    identifica los países más activos y genera datos para visualización.
    Úsala cuando el usuario pregunte por ubicación, países o distribución geográfica.

    Returns:
        Diccionario con distribución por zonas, países más activos y datos de mapa.
    """
    return _post("/analisis/geografico", {})


@tool
def tool_propagacion(message_id: str) -> dict:
    """
    Analiza cómo se propagó un mensaje específico a través del dataset.
    Mide alcance total, respuestas directas e indirectas, profundidad del hilo
    y velocidad media de respuesta. Úsala cuando el usuario mencione un ID
    de mensaje y quiera saber su propagación, alcance o difusión.

    Args:
        message_id: ID único del mensaje a analizar (ej: "msg_001", "abc123").

    Returns:
        Diccionario con métricas de propagación: alcance, profundidad, velocidad y timeline.
    """
    return _post("/analisis/propagacion", {"message_id": message_id})


@tool
def tool_semantico(query: str, n_results: int = 5, country: Optional[str] = None) -> dict:
    """
    Busca mensajes semánticamente similares a una consulta en lenguaje natural.
    Usa embeddings vectoriales (ChromaDB) para encontrar mensajes relevantes
    por significado, no por palabras exactas.
    Úsala cuando el usuario quiera encontrar mensajes sobre un tema específico,
    buscar opiniones similares, o explorar el dataset de forma semántica.

    Args:
        query:     Texto o pregunta en lenguaje natural para buscar.
        n_results: Número de mensajes similares a devolver (default: 5).
        country:   Filtrar resultados por país (opcional).

    Returns:
        Diccionario con lista de mensajes ordenados por similitud semántica.
    """
    payload: dict = {"query": query, "n_results": n_results}
    if country:
        payload["country"] = country
    return _post("/analisis/semantico", payload)


TOOLS = [tool_resumen, tool_geografico, tool_propagacion, tool_semantico]

TOOLS_BY_INTENT = {
    "resumen": tool_resumen,
    "geografico": tool_geografico,
    "propagacion": tool_propagacion,
    "semantico": tool_semantico,
}
