from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from . import config

router = APIRouter(prefix="/analisis", tags=["resumen"])

llm = ChatGoogleGenerativeAI(
    model=config.GOOGLE_MODEL,
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0.7
)


class ResumenRequest(BaseModel):
    conversations: list[dict] = Field(default_factory=list, description="Lista de conversaciones")
    tematica: Optional[str] = Field(None, description="Tema específico a analizar")


class Postura(BaseModel):
    postura: str
    porcentaje: float
    descripcion: str


class ResumenResponse(BaseModel):
    resumen: str
    tematica_principal: str
    posturas: list[Postura]
    palabras_clave: list[str]
    num_conversaciones: int


@router.post("/resumen", response_model=ResumenResponse)
async def analisis_resumen(request: ResumenRequest):
    conversations = request.conversations

    if not conversations:
        conversations = config.load_dataset_as_dicts()

    num_conversaciones = len(conversations)

    textos = [conv.get("text", conv.get("caption", "")) for conv in conversations if conv.get("text") or conv.get("caption")]
    textos_preview = textos[:50] if len(textos) > 50 else textos
    textos_str = "\n".join([f"- {t}" for t in textos_preview])

    prompt = f"""Eres un analistas de conversaciones digitales. Analiza las siguientes conversaciones y proporciona un resumen ejecutivo.

Conversaciones:
{textos_str}

{'Tema específico a analizar: ' + request.tematica if request.tematica else ''}

Proporciona un JSON con:
1. "resumen": Resumen ejecutivo de max 3 oraciones
2. "tematica_principal": Tema central identificado en una frase
3. "posturas": Array de 3 posturas con поля: postura (positiva/negativa/neutral), porcentaje (0-100), descripcion
4. "palabras_clave": Array de 5-10 palabras clave relevantes
5. "num_conversaciones": Número total de mensajes analizados

Responde SOLO con JSON válido, sin texto adicional."""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        import json
        result = json.loads(content.strip())

        return ResumenResponse(
            resumen=result.get("resumen", ""),
            tematica_principal=result.get("tematica_principal", ""),
            posturas=[Postura(**p) for p in result.get("posturas", [])],
            palabras_clave=result.get("palabras_clave", []),
            num_conversaciones=num_conversaciones
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar resumen: {str(e)}")