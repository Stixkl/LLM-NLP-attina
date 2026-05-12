from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from . import config
from data.loader import get_loader

router = APIRouter(prefix="/analisis", tags=["propagacion"])


class TimelineEntry(BaseModel):
    timestamp: str
    depth: int
    count: int


class PropagacionRequest(BaseModel):
    message_id: str = Field(..., description="ID del mensaje a analizar")
    conversaciones: list[dict] = Field(default_factory=list, description="Lista de conversaciones")


class PropagacionResponse(BaseModel):
    id_original: str
    alcance_total: int
    respuestas_directas: int
    respuestas_indirectas: int
    profundidad_maxima: int
    velocidad_media_minutos: float
    timeline: list[TimelineEntry]
    mensaje_original: str


def build_reply_tree(conversations: list[dict], root_id: str) -> dict[str, list[str]]:
    tree: dict[str, list[str]] = {}

    for conv in conversations:
        parent_id = conv.get("parentId") or conv.get("parent_id")
        msg_id = conv.get("id") or conv.get("message_id")

        if parent_id and msg_id:
            if parent_id not in tree:
                tree[parent_id] = []
            tree[parent_id].append(msg_id)

    return tree


def count_replies(tree: dict[str, list[str]], message_id: str, depth: int = 0) -> tuple[int, int, int, list[TimelineEntry]]:
    direct = 0
    indirect = 0
    max_depth = depth
    timeline_data: list[TimelineEntry] = []

    children = tree.get(message_id, [])

    for child_id in children:
        direct += 1
        sub_direct, sub_indirect, sub_depth, sub_timeline = count_replies(tree, child_id, depth + 1)
        indirect += sub_direct + sub_indirect
        max_depth = max(max_depth, sub_depth)
        timeline_data.extend(sub_timeline)

    if children:
        timeline_data.append(TimelineEntry(timestamp=f"depth_{depth}", depth=depth, count=len(children)))

    return direct, indirect, max_depth, timeline_data


@router.post("/propagacion", response_model=PropagacionResponse)
async def analisis_propagacion(request: PropagacionRequest):
    conversations = request.conversaciones

    if not conversations:
        try:
            loader = get_loader()
            conversations = loader.load_raw_json(config.SAMPLE_DATASET)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="No se proporcionaron conversaciones y no se encontró dataset")

    target_id = request.message_id

    original_message = None
    for conv in conversations:
        msg_id = conv.get("id") or conv.get("message_id")
        if msg_id == target_id:
            original_message = conv.get("text") or conv.get("caption", "")
            original_created = conv.get("createdAt") or conv.get("created_at")
            break

    if not original_message:
        raise HTTPException(status_code=404, detail=f"Mensaje con ID {target_id} no encontrado")

    tree = build_reply_tree(conversations, target_id)

    respuestas_directas, respuestas_indirectas, profundidad_maxima, timeline = count_replies(tree, target_id)

    alcance_total = respuestas_directas + respuestas_indirectas

    velocidad_media = 0.0
    if alcance_total > 0 and original_created:
        try:
            root_time = datetime.fromisoformat(original_created.replace("Z", "+00:00"))

            all_replies = []
            for conv in conversations:
                parent_id = conv.get("parentId") or conv.get("parent_id")
                if parent_id == target_id or parent_id in [c for c in tree.get(target_id, [])]:
                    reply_time_str = conv.get("createdAt") or conv.get("created_at")
                    if reply_time_str:
                        try:
                            reply_time = datetime.fromisoformat(reply_time_str.replace("Z", "+00:00"))
                            all_replies.append((reply_time - root_time).total_seconds() / 60)
                        except:
                            pass

            if all_replies:
                velocidad_media = sum(all_replies) / len(all_replies)
        except:
            velocidad_media = 0.0

    return PropagacionResponse(
        id_original=target_id,
        alcance_total=alcance_total,
        respuestas_directas=respuestas_directas,
        respuestas_indirectas=respuestas_indirectas,
        profundidad_maxima=profundidad_maxima,
        velocidad_media_minutos=round(velocidad_media, 2),
        timeline=timeline,
        mensaje_original=original_message[:200] + "..." if len(original_message) > 200 else original_message
    )