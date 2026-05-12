# Context & Role Definition - LLM-NLP-attina Project

---

## Role: Senior AI/ML Engineer

Eres un ingeniero senior especializado en desarrollo de sistemas basados en LLMs, agentes conversacionales y arquitecturas MCP (Model Context Protocol). Tu experiencia incluye:

- Diseño e implementación de pipelines de NLP/LLM
- Construcción de agentes con LangGraph y arquitecturas de tool-calling
- Desarrollo de microservicios REST con FastAPI
- Integración de LLMs (Gemini, GPT-4, Claude) en aplicaciones de producción
- Procesamiento y análisis de datos conversacionales a escala

---

## Project Overview

### What is Attina?

Attina es un sistema de análisis de conversaciones digitales. En la era digital, la capacidad de procesar y comprender grandes volúmenes de conversaciones en línea es crucial para la toma de decisiones estratégicas. Este proyecto simula un entorno donde se analiza datos sociales reales (publicaciones, comentarios, interacciones) para extraer información significativa.

### The Goal

Construir un **Agente Conversacional inteligente** que permita a usuarios no técnicos hacer preguntas en lenguaje natural sobre datos de conversaciones y recibir respuestas precisas generadas por LLMs, utilizando pipelines de análisis especializados (MCP services) ejecutados en backend.

Ejemplo: Un usuario pregunta "¿Cómo está el clima de la conversación?" y el agente ejecuta internamente el análisis de sentimientos, luego presenta el resultado en formato conversacional.

---

## Problem Statement

### The Challenge

El taller requiere construir un sistema completo que conste de:

1. **3 MCP Services (Microservicios)** que exponen análisis sobre conversaciones:
   - 2 servicios seleccionados por el equipo
   - 1 servicio obligatorio de propagación de mensajes

2. **1 Agente Conversacional** que:
   - Identifica la intención del usuario
   - Llama al MCP correcto según la pregunta
   - Genera respuestas en lenguaje natural usando los datos del MCP

3. **1 Interfaz de Demo** donde se pueda interactuar con el agente

### Análisis Disponibles (Select 2 + 1 Mandatory)

#### Servicios a Seleccionar:

1. **Topología de red**: Analiza la interacción y conexión entre comentarios (grafos, estructuras reply/thread)
2. **Resumen general**: Generación de resumen conciso sobre temática y posturas clave (requiere LLM)
3. **Análisis de sentimientos**: Polaridad (positivo/negativo/neutral) de comentarios
4. **Análisis de emociones**: Identificación de emociones evocadas (Felicidad, Furia, Angustia, etc.)
5. **Análisis de métricas**: Actores influyentes, posts con mayor impacto, likes, redes
6. **Análisis geográfico**: Ubicación de la conversación y visualización

#### Servicio Obligatorio:

7. **Análisis de Propagación**: A partir de un ID de mensaje, medir alcance, velocidad y presencia del mensaje original en respuestas/comentarios/hilos.

---

## Technical Stack

### Selected Technologies

| Component | Technology | Why |
|---|---|---|
| LLM | Google Gemini (API) | Cost-effective, good documentation, easy LangChain integration |
| Agent Framework | LangGraph | Explicit state management, graph-based control flow, advanced capabilities |
| MCP Services | FastAPI | Fast, automatic schema validation, LangChain tool compatibility |
| Demo Interface | Streamlit | Rapid prototyping, no frontend complexity needed |
| LLM Library | langchain-google-genai | Abstraction layer over Gemini API, facilitates tool-calling |
| Data Format | JSON | Standard for API responses and MCP communication |

### Dependencies (requirements.txt)
```
langgraph
langchain-core
langchain-google-genai
google-generativeai
fastapi
uvicorn
streamlit
pandas
python-dotenv
pydantic
httpx
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        UI (Streamlit)                          │
│                    User queries in natural language            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agent (LangGraph)                            │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────────────┐    │
│  │ Router  │───▶│ Tool Caller │───▶│ Response Generator   │    │
│  │(intent) │    │  (MCP exec) │    │  (Gemini + context)  │    │
│  └─────────┘    └─────────────┘    └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
         │                                           ▲
         │ Tools (3)                                 │
         ▼                                           │
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Services (FastAPI)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐          │
│  │  Resumen    │  │  Geográfico │  │   Propagación    │          │
│  │  General    │  │  (Select 1) │  │   (Mandatory)    │          │
│  └─────────────┘  └─────────────┘  └─────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data (JSON/CSV)                              │
│         conversations.json - Dataset de conversaciones          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dataset Schema

El dataset de conversaciones debe contener al menos los siguientes campos:

```json
{
  "post_id": "string - ID único del post",
  "user_id": "string - ID del usuario",
  "username": "string - Nombre de usuario",
  "text": "string - Contenido del mensaje/comentario",
  "timestamp": "ISO 8601 - Fecha y hora de publicación",
  "likes": "integer - Número de likes",
  "replies_count": "integer - Número de respuestas directas",
  "parent_id": "string|null - ID del mensaje padre (null si es post raíz)",
  "location": {
    "country": "string - País",
    "city": "string - Ciudad",
    "coordinates": {
      "lat": "float - Latitud",
      "lng": "float - Longitud"
    }
  }
}
```

---

## MCP Service Specifications

### MCP 1: Resumen General (SELECTED)
**Endpoint:** `POST /analisis/resumen`

**Input:**
```json
{
  "conversations": [...],
  "tematica": "string (opcional)"
}
```

**Output:**
```json
{
  "resumen": "string - Resumen ejecutivo de la conversación",
  "tematica_principal": "string - Tema central identificado",
  "posturas": [
    {"postura": "string", "porcentaje": 0.0, "descripcion": "string"}
  ],
  "palabras_clave": ["string"],
  "num_conversaciones": 0
}
```

**Implementation notes:**
- Use Gemini for synthesis with few-shot prompting
- Include tone/stance analysis
- Extract key topics and keywords

### MCP 2: Análisis Geográfico (SELECTED)
**Endpoint:** `POST /analisis/geografico`

**Input:**
```json
{
  "conversaciones": [...]
}
```

**Output:**
```json
{
  "distribucion": [
    {"region": "Norte", "porcentaje": 0.0, "count": 0},
    {"region": "Sur", "porcentaje": 0.0, "count": 0},
    {"region": "Este", "porcentaje": 0.0, "count": 0},
    {"region": "Oeste", "porcentaje": 0.0, "count": 0},
    {"region": "Centro", "porcentaje": 0.0, "count": 0}
  ],
  "paises_mas_activos": [
    {"pais": "string", "count": 0}
  ],
  "total_ubicaciones": 0,
  "mapa_data": [...]
}
```

**Implementation notes:**
- Group by cardinal direction (N/S/E/W) based on coordinates
- Calculate percentages
- Prepare data for potential visualization

### MCP 3: Propagación (OBLIGATORY)
**Endpoint:** `POST /analisis/propagacion`

**Input:**
```json
{
  "message_id": "string - ID del mensaje a analizar",
  "conversaciones": [...]
}
```

**Output:**
```json
{
  "id_original": "string",
  "alcance_total": 0,
  "respuestas_directas": 0,
  "respuestas_indirectas": 0,
  "profundidad_maxima": 0,
  "velocidad_media_minutos": 0.0,
  "timeline": [
    {"timestamp": "ISO 8601", "depth": 0, "count": 0}
  ],
  "mensaje_original": "string"
}
```

**Implementation notes:**
- Recursive tree traversal for replies
- Calculate spread velocity (time between original and replies)
- Measure thread depth and breadth

---

## Agent Design (LangGraph)

### State Schema
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    ultimo_analisis: Optional[dict]          # Last MCP result
    intencion: Optional[str]                  # Detected intent
    mcp_response: Optional[dict]              # MCP raw response
```

### Node Definitions

1. **router**: Analyzes user input and classifies intent
   - Intents: "resumen", "geografico", "propagacion", "seguimiento", "otro"
   - Uses Gemini with classification prompt

2. **call_mcp_resumen**: Calls `/analisis/resumen` endpoint
   - Parses conversations from state
   - Returns structured analysis

3. **call_mcp_geografico**: Calls `/analisis/geografico` endpoint
   - Sends conversation data
   - Returns geographic distribution

4. **call_mcp_propagacion**: Calls `/analisis/propagacion` endpoint
   - Receives message_id from user input
   - Returns propagation metrics

5. **respond**: Generates conversational response
   - Takes MCP output as context
   - Uses Gemini to produce friendly response
   - References previous context for follow-ups

### Graph Flow
```
User Input
    │
    ▼
┌────────┐     ┌──────────────┐
│ Router │────▶│ Intent: resumen │
└────────┘     └──────┬─────────┘
     │                 │
     │    ┌────────────┴───────┐
     │    ▼                    ▼
     │ ┌──────────┐   ┌──────────────┐
     │ │geografico│   │  propagacion │
     │ └──────────┘   └──────────────┘
     │
     ▼
┌─────────────────────────────┐
│        Respond              │
│   (Gemini + MCP context)    │
└─────────────────────────────┘
    │
    ▼
Final Response to User
```

---

## Evaluation Criteria

| Criterion | Weight | Description |
|---|---|---|
| MCP Functionality | 33% | All 3 services functional, fast, JSON-compliant |
| Analysis Quality | 33% | Precision and relevance. Sophisticated prompts, good data processing |
| Agent Intelligence | 33% | User intent identification, correct tool calling, natural conversation |
| Advanced Frameworks (Bonus) | 10% | Effective LangGraph usage with explicit state management |

---

## Interacciones de Demo Esperadas

| User Query | Agent Action |
|---|---|
| "Quiero saber cómo se ha propagado el mensaje con ID: 12345" | Call MCP Propagación |
| "¿Me puedes dar un resumen ejecutivo de la discusión sobre [tema]?" | Call MCP Resumen |
| "¿Cuál es la distribución geográfica de la conversación?" | Call MCP Geográfico |
| "¿El sentimiento es positivo?" | Reference previous response or re-call |
| "¿Cuál es el post que más impacto ha tenido?" | Call MCP with metric extraction |

---

## Constraints & Guidelines

### API Key Management
- `GOOGLE_API_KEY` must be stored in `.env` file
- `.env` must be in `.gitignore`
- Never commit API keys to repository

### Data Privacy
- If dataset contains real user data, anonymize before use
- Do not expose PII in API responses

### Code Conventions
- Use type hints for all function parameters and return values
- Follow existing patterns in the codebase
- Add docstrings to public functions
- Minimize comments (code should be self-explanatory)

### Testing
- Test each MCP endpoint before integration
- Use curl or Postman for manual verification
- Log which MCP was called for debugging

### Error Handling
- MCPs should return proper error JSON: `{"error": "string", "details": "string"}`
- Agent should handle MCP failures gracefully and inform the user

---

## Important Notes for AI Agents

1. **Start with MCP services** - Everything depends on these being functional first
2. **LangGraph requires understanding StateGraph** - Read documentation on state management before implementation
3. **The dataset must have `parent_id`** - Without this, propagation analysis cannot work
4. **Follow the commit-per-phase methodology** - Each phase must be verified before moving on
5. **Gemini prompts must be well-structured** - Use few-shot examples for consistent output formats
6. **The agent needs memory** - Use conversation history to handle follow-up questions

---

## File Structure

```
LLM-NLP-attina/
├── data/
│   ├── conversations.json          # Main dataset
│   └── sample_conversations.json   # Test dataset
├── services/                       # MCP Services (FastAPI)
│   ├── __init__.py
│   ├── config.py
│   ├── mcp_resumen.py
│   ├── mcp_geografico.py
│   └── mcp_propagacion.py
├── agent/                          # Agent (LangGraph)
│   ├── __init__.py
│   ├── graph.py
│   ├── tools.py
│   └── prompts.py
├── ui/                             # Interface (Streamlit)
│   └── app.py
├── docs/
│   ├── Plan-to-follow.md           # Execution plan
│   └── Context-roles.md            # This file
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

*Document version: 1.0*
*Last updated: 2026-05-11*
*Purpose: AI Agent onboarding and role definition*