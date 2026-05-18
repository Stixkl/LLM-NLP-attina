"""Interfaz Streamlit para el agente conversacional Attina."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.graph import AgentState, run_agent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Attina — Agente de Análisis",
    page_icon="📊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# CSS mínimo
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.mcp-badge {
    display:inline-block;
    padding:2px 10px;
    border-radius:12px;
    font-size:0.78rem;
    font-weight:600;
    margin-bottom:6px;
}
.badge-resumen    { background:#dbeafe; color:#1d4ed8; }
.badge-geografico { background:#dcfce7; color:#15803d; }
.badge-propagacion{ background:#fef3c7; color:#b45309; }
.badge-seguimiento{ background:#f3e8ff; color:#7c3aed; }
.badge-otro       { background:#f1f5f9; color:#64748b; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "agent_state" not in st.session_state:
    st.session_state.agent_state: AgentState | None = None

if "chat_history" not in st.session_state:
    # list of {"role": "user"|"assistant", "content": str, "intent": str|None}
    st.session_state.chat_history = []

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("Attina")
    st.caption("Agente de análisis de conversaciones digitales")
    st.divider()

    st.markdown("**Análisis disponibles:**")
    st.markdown("📝 **Resumen** — Síntesis y posturas clave")
    st.markdown("📍 **Geográfico** — Distribución por regiones")
    st.markdown("🔁 **Propagación** — Alcance de un mensaje")
    st.divider()

    st.markdown("**Preguntas de ejemplo:**")
    examples = [
        "¿Puedes darme un resumen de la conversación?",
        "¿De qué países vienen los mensajes?",
        "¿Cómo se propagó el mensaje msg_001?",
        "¿Cuál es la distribución geográfica?",
        "Dame un resumen sobre tecnología",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["_pending_input"] = ex

    st.divider()
    st.session_state.debug_mode = st.toggle("🔧 Modo debug (ver MCP data)", value=st.session_state.debug_mode)

    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.agent_state = None
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Attina — Agente Conversacional")
st.caption("Pregúntame sobre las conversaciones del dataset en lenguaje natural.")
st.divider()

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
INTENT_LABELS = {
    "resumen": ("badge-resumen", "📝 Resumen"),
    "geografico": ("badge-geografico", "📍 Geográfico"),
    "propagacion": ("badge-propagacion", "🔁 Propagación"),
    "seguimiento": ("badge-seguimiento", "💬 Seguimiento"),
    "otro": ("badge-otro", "❓ General"),
}

for turn in st.session_state.chat_history:
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.write(turn["content"])
    else:
        with st.chat_message("assistant", avatar="📊"):
            intent = turn.get("intent")
            if intent and intent in INTENT_LABELS:
                css_class, label = INTENT_LABELS[intent]
                st.markdown(
                    f'<span class="mcp-badge {css_class}">{label}</span>',
                    unsafe_allow_html=True,
                )
            st.write(turn["content"])

            if st.session_state.debug_mode and turn.get("mcp_data"):
                with st.expander("🔍 Datos crudos del MCP"):
                    st.json(turn["mcp_data"])

# ---------------------------------------------------------------------------
# Handle pending input from sidebar buttons
# ---------------------------------------------------------------------------
pending = st.session_state.pop("_pending_input", None)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input("Escribe tu pregunta aquí…") or pending

if user_input:
    # Show user message immediately
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Run agent
    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("Analizando…"):
            try:
                response_text, new_state = run_agent(
                    user_input,
                    st.session_state.agent_state,
                )
                st.session_state.agent_state = new_state

                intent = new_state.get("intencion")
                mcp_data = new_state.get("mcp_response")

                if intent and intent in INTENT_LABELS:
                    css_class, label = INTENT_LABELS[intent]
                    st.markdown(
                        f'<span class="mcp-badge {css_class}">{label}</span>',
                        unsafe_allow_html=True,
                    )

                st.write(response_text)

                if st.session_state.debug_mode and mcp_data:
                    with st.expander("🔍 Datos crudos del MCP"):
                        st.json(mcp_data)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_text,
                    "intent": intent,
                    "mcp_data": mcp_data,
                })

            except Exception as exc:
                error_msg = f"⚠️ Error inesperado: {exc}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "intent": None,
                    "mcp_data": None,
                })
