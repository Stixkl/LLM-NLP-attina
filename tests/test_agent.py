"""Tests unitarios para el agente Attina — Fase 3."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.graph import _extract_message_id, route_by_intent, AgentState


# ---------------------------------------------------------------------------
# Tests: _extract_message_id
# ---------------------------------------------------------------------------

class TestExtractMessageId:
    def test_explicit_id_prefix(self):
        assert _extract_message_id("propaga el mensaje ID: msg_001") == "msg_001"

    def test_msg_prefix(self):
        assert _extract_message_id("cómo se propagó msg_042") == "msg_042"

    def test_post_prefix(self):
        assert _extract_message_id("analiza el post post_99") == "post_99"

    def test_bare_id(self):
        assert _extract_message_id("propaga abc123def") == "abc123def"

    def test_no_id_returns_none(self):
        result = _extract_message_id("dame un resumen de la conversación")
        assert result is None

    def test_id_with_colon(self):
        assert _extract_message_id("mensaje ID:xyz789") == "xyz789"


# ---------------------------------------------------------------------------
# Tests: route_by_intent
# ---------------------------------------------------------------------------

class TestRouteByIntent:
    def _make_state(self, intencion: str) -> AgentState:
        return AgentState(
            messages=[],
            ultimo_analisis=None,
            intencion=intencion,
            mcp_response=None,
            error=None,
        )

    def test_routes_resumen(self):
        assert route_by_intent(self._make_state("resumen")) == "call_resumen"

    def test_routes_geografico(self):
        assert route_by_intent(self._make_state("geografico")) == "call_geografico"

    def test_routes_propagacion(self):
        assert route_by_intent(self._make_state("propagacion")) == "call_propagacion"

    def test_routes_seguimiento_to_respond(self):
        assert route_by_intent(self._make_state("seguimiento")) == "respond"

    def test_routes_otro_to_respond(self):
        assert route_by_intent(self._make_state("otro")) == "respond"

    def test_routes_unknown_to_respond(self):
        assert route_by_intent(self._make_state("unknown_intent")) == "respond"


# ---------------------------------------------------------------------------
# Tests: tool calls (mocked HTTP)
# ---------------------------------------------------------------------------

class TestToolCalls:
    @patch("agent.tools.httpx.Client")
    def test_tool_resumen_calls_correct_endpoint(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "resumen": "test",
            "tematica_principal": "tecnología",
            "posturas": [],
            "palabras_clave": ["ai"],
            "num_conversaciones": 5,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        from agent.tools import tool_resumen
        result = tool_resumen.invoke({"tematica": "tecnología"})

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/analisis/resumen" in call_args[0][0]
        assert result["tematica_principal"] == "tecnología"

    @patch("agent.tools.httpx.Client")
    def test_tool_geografico_calls_correct_endpoint(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "distribucion": [],
            "paises_mas_activos": [],
            "total_ubicaciones": 0,
            "mapa_data": [],
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        from agent.tools import tool_geografico
        tool_geografico.invoke({})

        call_args = mock_client.post.call_args
        assert "/analisis/geografico" in call_args[0][0]

    @patch("agent.tools.httpx.Client")
    def test_tool_propagacion_sends_message_id(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id_original": "msg_001",
            "alcance_total": 3,
            "respuestas_directas": 2,
            "respuestas_indirectas": 1,
            "profundidad_maxima": 2,
            "velocidad_media_minutos": 5.0,
            "timeline": [],
            "mensaje_original": "Hola mundo",
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        from agent.tools import tool_propagacion
        result = tool_propagacion.invoke({"message_id": "msg_001"})

        call_args = mock_client.post.call_args
        assert "/analisis/propagacion" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["message_id"] == "msg_001"
        assert result["alcance_total"] == 3


# ---------------------------------------------------------------------------
# Tests: node_call_propagacion — missing ID
# ---------------------------------------------------------------------------

class TestNodePropagacion:
    def test_returns_error_when_no_id_found(self):
        from langchain_core.messages import HumanMessage
        from agent.graph import node_call_propagacion

        state = AgentState(
            messages=[HumanMessage(content="dime la propagación")],
            ultimo_analisis=None,
            intencion="propagacion",
            mcp_response=None,
            error=None,
        )
        result = node_call_propagacion(state)
        assert result["error"] is not None
        assert result["mcp_response"] is None