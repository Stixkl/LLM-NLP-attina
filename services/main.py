from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import config
from .mcp_resumen import router as resumen_router
from .mcp_geografico import router as geografico_router
from .mcp_propagacion import router as propagacion_router
from .mcp_semantico import router as semantico_router

app = FastAPI(
    title="Attina MCP Services",
    description="Microservicios de análisis de conversaciones",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    from data.vector_store import get_stats
    chroma_stats = get_stats()
    return {
        "status": "ok",
        "service": "Attina MCP",
        "model": config.GOOGLE_MODEL,
        "env": config.APP_ENV,
        "chromadb": chroma_stats,
    }


@app.get("/")
def root():
    return {
        "message": "Attina MCP Services API",
        "docs": "/docs",
        "endpoints": {
            "resumen": "/analisis/resumen",
            "geografico": "/analisis/geografico",
            "propagacion": "/analisis/propagacion",
            "semantico": "/analisis/semantico",
        },
    }


app.include_router(resumen_router)
app.include_router(geografico_router)
app.include_router(propagacion_router)
app.include_router(semantico_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.MCP_HOST,
        port=config.MCP_PORT,
        reload=True
    )