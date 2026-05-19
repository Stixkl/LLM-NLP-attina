import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))

APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DATA_DIR = BASE_DIR / "data"
DEFAULT_DATASET = "conversations.parquet"
SAMPLE_DATASET = "sample_conversations.json"

def load_dataset_as_dicts() -> list[dict]:
    """Carga el dataset disponible como lista de dicts.
    Busca en orden: DEFAULT_DATASET (parquet), cualquier .parquet, SAMPLE_DATASET (json).
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(BASE_DIR))
    from data.loader import get_loader

    loader = get_loader()

    # 1. Parquet por defecto
    if (DATA_DIR / DEFAULT_DATASET).exists():
        msgs = loader.load_parquet(DEFAULT_DATASET)
        return [{"text": m.text, "id": m.id, "parentId": m.parent_id,
                 "latitude": m.location.latitude if m.location else None,
                 "longitude": m.location.longitude if m.location else None,
                 "country": m.location.country if m.location else None,
                 "createdAt": m.created_at.isoformat()} for m in msgs]

    # 2. Cualquier parquet en data/
    parquets = sorted(DATA_DIR.glob("*.parquet"))
    if parquets:
        msgs = loader.load_parquet(parquets[0].name)
        return [{"text": m.text, "id": m.id, "parentId": m.parent_id,
                 "latitude": m.location.latitude if m.location else None,
                 "longitude": m.location.longitude if m.location else None,
                 "country": m.location.country if m.location else None,
                 "createdAt": m.created_at.isoformat()} for m in msgs]

    # 3. JSON de muestra
    raw = loader.load_raw_json(SAMPLE_DATASET)
    return raw