# Plan de Ejecución - Agente Conversacional Attina

## 1. Contexto del Proyecto

Este proyecto simula un entorno de análisis de datos sociales donde los participantes deben construir un **Agente Conversacional inteligente** capaz de interactuar con microservicios especializados (MCP) para el análisis de conversaciones digitales.

El objetivo final es crear un sistema donde el usuario pueda preguntar de forma natural (ej: "¿Cómo está el clima de la conversación?") y el agente ejecute el pipeline de procesamiento analítico más adecuado para responder.

---

## 2. Tecnologías Seleccionadas

| Componente | Tecnología | Justificación |
|---|---|---|
| **LLM** | Google Gemini (API) | Balance costo-rendimiento, documentación clara, fácil integración con LangChain |
| **Agente** | LangGraph | Mayor control sobre flujo conversacional, manejo de estado explícito, puntos extra en evaluación |
| **MCP Services** | FastAPI | Framework rápido, validación automática de schemas, compatible con LangChain tools |
| **Interfaz Demo** | Streamlit | Prototipado rápido, sin necesidad de frontend complejo, ideal para demos |
| **Librería LLM** | langchain-google-genai | Abstracción sobre la API de Gemini, facilita tool-calling y chat models |
| **Dataset** | Provisto por el equipo | Los datos de conversaciones reales ya se encuentran disponibles |

### Dependencias principales
```
langgraph
fastapi
uvicorn
streamlit
langchain-google-genai
google-generativeai
pandas
```

---

## 3. Análisis a Implementar

Se implementarán **3 servicios MCP** como se especifica en el enunciado del taller:

### 3.1 Servicios Seleccionados

| # | Análisis | Descripción | Entrada | Salida |
|---|---|---|---|---|
| 1 | **Resumen General** | Generación de un resumen conciso sobre la temática principal y las posturas clave | Dataset de conversaciones | Resumen en texto plano + keywords |
| 2 | **Análisis Geográfico** | Ubicación de dónde está ocurriendo la conversación | Dataset con datos de ubicación | Distribución por zonas cardinales + visualización |

### 3.2 Servicio Obligatorio

| # | Análisis | Descripción | Entrada | Salida |
|---|---|---|---|---|
| 3 | **Propagación** | Mide alcance, velocidad y presencia del mensaje original en respuestas/comentarios | ID de mensaje específico | Alcance, tiempo medio de respuesta, profundidad de thread |

---

## 4. Estructura del Proyecto

```
LLM-NLP-attina/
├── data/                          # Dataset y utilidades de carga
│   ├── __init__.py                 # Exports: Message, Author, Location, DataLoader, get_loader
│   ├── loader.py                   # DataLoader con cache y stream batching
│   ├── schema.py                   # Modelos: Message, Author, Location, MessageType, SocialSource
│   ├── sample_conversations.json   # Dataset de prueba (13 mensajes multilingüe)
│   └── README.md                   # Documentación del schema
│
├── services/                       # Microservicios MCP (FastAPI)
│   ├── __init__.py
│   ├── config.py                   # Configuraciones compartidas
│   ├── mcp_resumen.py              # Servicio: Resumen General
│   ├── mcp_geografico.py           # Servicio: Análisis Geográfico
│   └── mcp_propagacion.py          # Servicio: Propagación (obligatorio)
│
├── agent/                          # Agente conversacional (LangGraph)
│   ├── __init__.py
│   ├── graph.py                    # Definición del grafo (estados, nodos, aristas)
│   ├── tools.py                    # Definición de tools (MCP clients)
│   └── prompts.py                  # Prompts del agente
│
├── ui/                             # Interfaz de demostración
│   └── app.py                      # Aplicación Streamlit
│
├── tests/                          # Tests unitarios
│   └── __init__.py
│
├── docs/                           # Documentación
│   ├── Plan-to-follow.md           # Plan de ejecución
│   └── Context-roles.md            # Contexto y rol para agentes IA
│
├── .env                            # Variables de entorno (GOOGLE_API_KEY)
├── .gitignore                      # Ignora .env, data/, __pycache__/
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Documentación principal
```

---

## 5. Plan de Ejecución

### Fase 1: Configuración del Entorno ✅
**Objetivo:** Preparar la estructura base y dependencias del proyecto.

- [x] Crear estructura de carpetas (`services/`, `agent/`, `data/`, `ui/`, `tests/`)
- [x] Crear `requirements.txt` con todas las dependencias
- [x] Crear archivo `.env` con `GOOGLE_API_KEY` y configuración del modelo
- [x] Colocar dataset de conversaciones en `data/` (pendiente: archivo del usuario)
- [x] Crear `data/sample_conversations.json` como dataset de prueba (13 mensajes de ejemplo)
- [x] Crear `data/schema.py` con modelos: Message, Author, Location, MessageType, SocialSource
- [x] Crear `data/loader.py` con DataLoader para cargar datasets JSON
- [x] Documentar la estructura del dataset (campos completos del esquema de Attina)

### Fase 2: Desarrollo de los 3 MCP Services ✅ COMPLETADA
**Objetivo:** Construir 3 endpoints REST funcionales que procesen el dataset y devuelvan JSON.

| Servicio | Endpoint | Descripción técnica | Estado |
|---|---|---|---|
| MCP Resumen | `POST /analisis/resumen` | Usa Gemini para síntesis de conversaciones. Prompt de few-shot para formato consistente. | ✅ |
| MCP Geográfico | `POST /analisis/geografico` | Procesa campo de ubicación, agrupa por zonas cardinales (Norte/Sur/Este/Oeste), calcula distribución porcentual. | ✅ |
| MCP Propagación | `POST /analisis/propagacion` | Recibe message_id, busca replies recursivamente, calcula alcance (directos + indirectos), velocidad media, profundidad. | ✅ |

**Verificación:** Cada endpoint responde correctamente con curl:
```bash
curl -X POST http://localhost:8000/analisis/resumen -H "Content-Type: application/json" -d "{}"
curl -X POST http://localhost:8000/analisis/geografico -H "Content-Type: application/json" -d "{}"
curl -X POST http://localhost:8000/analisis/propagacion -H "Content-Type: application/json" -d "{\"message_id\": \"msg_001\"}"
```

### Fase 3: Construcción del Agente con LangGraph
**Objetivo:** Crear un agente que identifique la intención del usuario y llame al MCP correcto.

#### 3.1 Definición de Tools
- Definir 3 LangChain Tools que invoquen los endpoints de los MCP
- Crear JSON schemas descriptivos para que Gemini sepa qué parámetros necesita cada tool

#### 3.2 Construcción del Grafo
```
Estado: {"messages": [], "ultimo_analisis": None, "intencion": None}

Nodos:
  - router: Clasifica la intención del usuario (resumen / geografico / propagacion / seguimiento)
  - call_mcp_resumen: Invoca el servicio de resumen
  - call_mcp_geografico: Invoca el servicio de análisis geográfico
  - call_mcp_propagacion: Invoca el servicio de propagación
  - respond: Genera respuesta conversacional usando Gemini con los datos del MCP

Aristas condicionales:
  - router -> nodo según intención
  - nodo_mcp -> respond
```

#### 3.3 Prompts
- Prompt del router: Clasificador de intención con ejemplos de preguntas
- Prompt del respond: Genera respuesta amigable con base en los datos del análisis

### Fase 4: Interfaz Streamlit
**Objetivo:** Demo funcional donde el usuario pueda conversar con el agente.

- [ ] Campo de texto para input del usuario
- [ ] Llamada al agente con la pregunta
- [ ] Display de la respuesta del agente
- [ ] (Opcional) Mostrar logs de qué MCP se llamó y qué datos devolvió
- [ ] Historial de conversación en sesión

### Fase 5: Integración y Pruebas
**Objetivo:** Verificar que todo funcione end-to-end con las interacciones de ejemplo.

- [ ] Probar: "Quiero saber cómo se ha propagado el mensaje con ID: 12345"
- [ ] Probar: "¿Me puedes dar un resumen ejecutivo de la discusión?"
- [ ] Probar: "¿Cuál es la distribución geográfica de la conversación?"
- [ ] Probar: "¿El sentimiento es positivo?" (seguimiento de contexto)

---

## 6. Metodología de Trabajo

### Flujo de desarrollo
```
1. Crear/actualizar estructura de carpetas
2. Implementar código de la fase
3. Probar manualmente (curl / streamlit dev)
4. Commit con mensaje descriptivo
5. Avanzar a la siguiente fase
```

### Criterios de aceptación por fase
- **Fase 1:** Se puede hacer `pip install -r requirements.txt` sin errores
- **Fase 2:** Los 3 endpoints responden con JSON válido en Postman/curl
- **Fase 3:** El agente routing funciona para las 4 intenciones (resumen, geo, propagacion, seguimiento)
- **Fase 4:** La interfaz Streamlit permite conversar con el agente sin errores
- **Fase 5:** Las 4 interacciones de ejemplo funcionan correctamente

---

## 7. Interacciones de Demo (Objetivo Final)

| Pregunta del Usuario | Acción del Agente |
|---|---|
| "Quiero saber cómo se ha propagado el mensaje con ID: 12345" | Llama a MCP Propagación con ese ID |
| "¿Me puedes dar un resumen ejecutivo de la discusión sobre [tema]?" | Llama a MCP Resumen |
| "¿Cuál es la distribución geográfica de la conversación?" | Llama a MCP Geográfico |
| "¿El sentimiento es positivo?" | Recuerda respuesta anterior o re-ejecuta el servicio |

---

## 8. Notas Importantes

- La **Fase 2 (MCPs)** es la más crítica: todo lo demás depende de que estos servicios funcionen
- Para **LangGraph**, es necesario entender bien el concepto de `StateGraph` con estados mutables
- El dataset debe tener al menos los campos: `post_id`, `user_id`, `text`, `timestamp`, `parent_id` (para propagación)
- La `GOOGLE_API_KEY` debe estar en `.env` y nunca debe subirse al repositorio (agregar a `.gitignore`)