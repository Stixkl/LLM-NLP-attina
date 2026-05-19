"""Grafo LangGraph del agente conversacional Attina."""

import re
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services import config
from agent.prompts import (
    FOLLOWUP_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    RESPOND_FALLBACK_PROMPT,
    RESPOND_SYSTEM_PROMPT,
    ROUTER_HUMAN_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from agent.tools import TOOLS_BY_INTENT, tool_geografico, tool_propagacion, tool_resumen, tool_semantico
from rag.retriever import build_context_string, retrieve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    ultimo_analisis: Optional[dict]   # Último resultado del MCP
    intencion: Optional[str]          # Intención clasificada
    mcp_response: Optional[dict]      # Respuesta cruda del MCP
    error: Optional[str]              # Error si ocurrió


# ---------------------------------------------------------------------------
# LLM instances
# ---------------------------------------------------------------------------

def _build_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GOOGLE_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Helper: extract message_id from user text
# ---------------------------------------------------------------------------

# Matches: "ID: abc123", "msg_042", "post_99", "message xyz"
_ID_WITH_PREFIX = re.compile(
    r'\b(?:id|ID|message)[:\s]+([A-Za-z0-9_-]+)',
    re.IGNORECASE,
)
# Matches tokens that look like IDs: word_digits or alphanumeric ≥6 chars
# Excludes common Spanish stopwords via negative lookahead
_STOPWORDS = {
    "cómo", "como", "dame", "dime", "quiero", "saber", "tenido", "propaga",
    "propago", "mensaje", "resumen", "analiza", "analisis", "geografico",
    "distribución", "distribucion", "conversación", "conversacion",
}
_STRUCTURED_ID = re.compile(r'\b([A-Za-z]+[_-]\d+)\b')          # msg_001, post_42
_LONG_ALPHANUM  = re.compile(r'\b([A-Za-z0-9]{8,})\b')           # bare alphanumeric ≥8


def _extract_message_id(text: str) -> Optional[str]:
    # 1. Explicit prefix: "ID: abc", "message: xyz"
    match = _ID_WITH_PREFIX.search(text)
    if match:
        return match.group(1)

    # 2. Structured IDs: word_digits pattern (msg_001, post_99)
    match = _STRUCTURED_ID.search(text)
    if match:
        return match.group(1)

    # 3. Long alphanumeric token not in stopwords
    for match in _LONG_ALPHANUM.finditer(text):
        token = match.group(1)
        if token.lower() not in _STOPWORDS:
            return token

    return None


# ---------------------------------------------------------------------------
# Helper: build compact conversation history string
# ---------------------------------------------------------------------------

def _history_str(messages: list, max_turns: int = 6) -> str:
    recent = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
    parts = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            parts.append(f"Usuario: {msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"Attina: {msg.content}")
    return "\n".join(parts) if parts else "Sin historial previo."


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_router(state: AgentState) -> AgentState:
    """Clasifica la intención del último mensaje del usuario."""
    llm = _build_llm(temperature=0.0)

    user_input = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    prompt = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=ROUTER_HUMAN_PROMPT.format(user_input=user_input)),
    ]

    response = llm.invoke(prompt)
    intencion = response.content.strip().lower().split()[0]

    valid = {"resumen", "geografico", "propagacion", "semantico", "seguimiento", "otro"}
    if intencion not in valid:
        intencion = "otro"

    return {**state, "intencion": intencion, "error": None}


def node_call_resumen(state: AgentState) -> AgentState:
    """Llama al MCP de resumen general."""
    user_input = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    tematica = None
    for kw in ["sobre", "acerca de", "respecto a", "de", "about"]:
        if kw in user_input.lower():
            parts = user_input.lower().split(kw, 1)
            if len(parts) > 1 and parts[1].strip():
                tematica = parts[1].strip().rstrip("?.,!")
                break

    try:
        result = tool_resumen.invoke({"tematica": tematica})
        return {**state, "mcp_response": result, "ultimo_analisis": result, "error": None}
    except Exception as exc:
        return {**state, "mcp_response": None, "error": str(exc)}


def node_call_geografico(state: AgentState) -> AgentState:
    """Llama al MCP de análisis geográfico."""
    try:
        result = tool_geografico.invoke({})
        return {**state, "mcp_response": result, "ultimo_analisis": result, "error": None}
    except Exception as exc:
        return {**state, "mcp_response": None, "error": str(exc)}


def node_call_propagacion(state: AgentState) -> AgentState:
    """Llama al MCP de propagación, extrayendo el message_id del input del usuario."""
    user_input = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    message_id = _extract_message_id(user_input)

    if not message_id:
        return {
            **state,
            "mcp_response": None,
            "error": "No pude identificar el ID del mensaje. Por favor, indícalo claramente (ej: 'propaga el mensaje msg_001').",
        }

    try:
        result = tool_propagacion.invoke({"message_id": message_id})
        return {**state, "mcp_response": result, "ultimo_analisis": result, "error": None}
    except Exception as exc:
        error_msg = str(exc)
        if "404" in error_msg:
            error_msg = f"No encontré el mensaje con ID '{message_id}' en el dataset."
        return {**state, "mcp_response": None, "error": error_msg}


def node_call_semantico(state: AgentState) -> AgentState:
    """Llama al MCP de búsqueda semántica usando ChromaDB."""
    user_input = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    try:
        result = tool_semantico.invoke({"query": user_input, "n_results": 5})
        return {**state, "mcp_response": result, "ultimo_analisis": result, "error": None}
    except Exception as exc:
        error_msg = str(exc)
        if "500" in error_msg or "collection" in error_msg.lower():
            error_msg = "La base de datos vectorial no tiene datos indexados. Ejecuta primero: python -m data.indexer"
        return {**state, "mcp_response": None, "error": error_msg}


def _get_user_input(state: AgentState) -> str:
    """Extrae el último mensaje del usuario del estado."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def node_respond(state: AgentState) -> AgentState:
    """Genera una respuesta usando RAG: recupera fragmentos relevantes y Gemini los sintetiza."""
    llm = _build_llm(temperature=0.7)
    intencion = state.get("intencion", "otro")
    error = state.get("error")
    mcp_response = state.get("mcp_response")
    historial = _history_str(state["messages"])
    user_input = _get_user_input(state)

    # --- Error path ---
    if error:
        response = llm.invoke([
            SystemMessage(content=(
                f"Eres Attina, asistente de análisis de conversaciones. "
                f"Informa al usuario de este problema de forma amigable: {error}. "
                f"Sugiere cómo solucionarlo si es posible."
            )),
            HumanMessage(content="Informa del error"),
        ])
        return {**state, "messages": [AIMessage(content=response.content)]}

    # --- Fallback / otro: RAG sobre pregunta abierta ---
    if intencion == "otro" or mcp_response is None:
        fragments = _rag_retrieve(user_input)
        if fragments:
            context = build_context_string(fragments)
            response = llm.invoke([
                SystemMessage(content=RAG_SYSTEM_PROMPT.format(
                    user_input=user_input,
                    context=context,
                    historial=historial,
                )),
                HumanMessage(content=user_input),
            ])
        else:
            response = llm.invoke([
                SystemMessage(content=RESPOND_FALLBACK_PROMPT.format(historial=historial)),
                HumanMessage(content="Responde al usuario"),
            ])
        return {**state, "messages": [AIMessage(content=response.content)]}

    # --- Follow-up: RAG enriquecido con el análisis previo ---
    if intencion == "seguimiento":
        ultimo = state.get("ultimo_analisis")
        fragments = _rag_retrieve(user_input)
        context = build_context_string(fragments) if fragments else "(Sin fragmentos adicionales)"

        if not ultimo and not fragments:
            response = llm.invoke([
                SystemMessage(content=(
                    "Eres Attina. No hay análisis previo ni contexto disponible. "
                    "Pide al usuario que realice primero un análisis (resumen, geográfico, propagación o búsqueda semántica)."
                )),
                HumanMessage(content="Responde"),
            ])
        else:
            datos_previos = json.dumps(ultimo, ensure_ascii=False, indent=2) if ultimo else "(Sin datos previos)"
            combined_prompt = (
                FOLLOWUP_SYSTEM_PROMPT.format(
                    tipo_analisis=state.get("intencion", "previo"),
                    datos_mcp=datos_previos,
                    historial=historial,
                )
                + f"\n\nFRAGMENTOS ADICIONALES RECUPERADOS:\n{context}"
            )
            response = llm.invoke([
                SystemMessage(content=combined_prompt),
                HumanMessage(content=user_input),
            ])
        return {**state, "messages": [AIMessage(content=response.content)]}

    # --- MCP response enriquecida con RAG ---
    # La query RAG combina la pregunta del usuario con el resultado del MCP
    # para recuperar fragmentos que complementen el análisis estructurado.
    rag_query = f"{user_input} {_mcp_summary_for_rag(mcp_response, intencion)}"
    fragments = _rag_retrieve(rag_query)
    context = build_context_string(fragments) if fragments else "(Sin fragmentos adicionales)"

    combined_prompt = (
        RESPOND_SYSTEM_PROMPT.format(
            tipo_analisis=intencion,
            datos_mcp=json.dumps(mcp_response, ensure_ascii=False, indent=2),
            historial=historial,
        )
        + f"\n\nFRAGMENTOS REALES RECUPERADOS PARA ENRIQUECER LA RESPUESTA:\n{context}"
        + "\n\nCita ejemplos reales de los fragmentos al presentar los resultados del análisis."
    )

    response = llm.invoke([
        SystemMessage(content=combined_prompt),
        HumanMessage(content="Genera la respuesta para el usuario basándote en los datos del análisis y los fragmentos recuperados."),
    ])

    return {**state, "messages": [AIMessage(content=response.content)]}


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def _rag_retrieve(query: str) -> list:
    """Recupera fragmentos con manejo de errores. Retorna lista vacía si ChromaDB no está listo."""
    if not query or not query.strip():
        return []
    try:
        return retrieve(query, k_conversations=4, k_documents=3, min_score=0.25)
    except Exception as exc:
        logger.warning("RAG retrieve falló (ChromaDB no disponible): %s", exc)
        return []


def _mcp_summary_for_rag(mcp_response: Optional[dict], intencion: str) -> str:
    """Extrae palabras clave del resultado MCP para enriquecer la query RAG."""
    if not mcp_response:
        return ""
    try:
        if intencion == "resumen":
            tematica = mcp_response.get("tematica_principal", "")
            keywords = " ".join(mcp_response.get("palabras_clave", [])[:5])
            return f"{tematica} {keywords}"
        elif intencion == "geografico":
            paises = " ".join(
                p.get("pais", "") for p in mcp_response.get("paises_mas_activos", [])[:3]
            )
            return paises
        elif intencion == "propagacion":
            return mcp_response.get("mensaje_original", "")[:200]
        elif intencion == "semantico":
            results = mcp_response.get("results", [])
            return " ".join(r.get("text", "")[:50] for r in results[:3])
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Conditional edge: router → next node
# ---------------------------------------------------------------------------

def route_by_intent(state: AgentState) -> str:
    intencion = state.get("intencion", "otro")
    routing = {
        "resumen": "call_resumen",
        "geografico": "call_geografico",
        "propagacion": "call_propagacion",
        "semantico": "call_semantico",
        "seguimiento": "respond",
        "otro": "respond",
    }
    return routing.get(intencion, "respond")


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router", node_router)
    graph.add_node("call_resumen", node_call_resumen)
    graph.add_node("call_geografico", node_call_geografico)
    graph.add_node("call_propagacion", node_call_propagacion)
    graph.add_node("call_semantico", node_call_semantico)
    graph.add_node("respond", node_respond)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "call_resumen": "call_resumen",
            "call_geografico": "call_geografico",
            "call_propagacion": "call_propagacion",
            "call_semantico": "call_semantico",
            "respond": "respond",
        },
    )

    graph.add_edge("call_resumen", "respond")
    graph.add_edge("call_geografico", "respond")
    graph.add_edge("call_propagacion", "respond")
    graph.add_edge("call_semantico", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


# Singleton compilado para reutilizar en UI y tests
attina_graph = build_graph()


# ---------------------------------------------------------------------------
# Public API: run one conversational turn
# ---------------------------------------------------------------------------

def run_agent(user_input: str, state: Optional[AgentState] = None) -> tuple[str, AgentState]:
    """
    Ejecuta un turno conversacional del agente Attina.

    Args:
        user_input: Mensaje del usuario en lenguaje natural.
        state:      Estado previo de la conversación (None para nueva sesión).

    Returns:
        Tupla (respuesta_texto, nuevo_estado).
    """
    if state is None:
        state = AgentState(
            messages=[],
            ultimo_analisis=None,
            intencion=None,
            mcp_response=None,
            error=None,
        )

    input_state = {
        **state,
        "messages": state["messages"] + [HumanMessage(content=user_input)],
    }

    result = attina_graph.invoke(input_state)

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    response_text = ai_messages[-1].content if ai_messages else "No se pudo generar una respuesta."

    return response_text, result
