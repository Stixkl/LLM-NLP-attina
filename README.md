# LLM-NLP-attina

Agente Conversacional inteligente para análisis de conversaciones digitales mediante servicios MCP.

## Estructura del Proyecto

```
LLM-NLP-attina/
├── data/                  # Datasets y utilidades de carga
│   ├── loader.py          # DataLoader para cargar datasets JSON
│   ├── schema.py          # Modelos (Message, Author, Location)
│   └── sample_conversations.json  # Dataset de prueba
├── services/              # Microservicios MCP (FastAPI)
├── agent/                 # Agente conversacional (LangGraph)
├── ui/                    # Interfaz de demostración (Streamlit)
├── tests/                # Tests unitarios
├── docs/                  # Documentación
├── requirements.txt      # Dependencias Python
├── .env                   # Variables de entorno
└── .gitignore
```

## Quick Start

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Configurar API Key
El archivo `.env` ya contiene la configuración de Google Gemini. Verifica que la key sea correcta.

### 3. Cargar dataset
Coloca tu archivo JSON en `data/conversations.json`.

### 4. Ejecutar servicios MCP
```bash
cd services && uvicorn main:app --reload
```

### 5. Ejecutar interfaz
```bash
cd ui && streamlit run app.py
```

## Análisis Disponibles

| Análisis | Endpoint | Descripción |
|---|---|---|
| Resumen General | `POST /analisis/resumen` | Síntesis de conversaciones con Gemini |
| Análisis Geográfico | `POST /analisis/geografico` | Distribución por regiones cardinales |
| Propagación | `POST /analisis/propagacion` | Alcance de un mensaje en la red |

## Tecnologías

- **LLM:** Google Gemini (API)
- **Agente:** LangGraph
- **MCP Services:** FastAPI
- **UI:** Streamlit
- **Librería:** langchain-google-genai