"""Script de indexación: carga el dataset y lo indexa en ChromaDB.

Uso:
    python -m data.indexer                          # indexa conversations.parquet (default)
    python -m data.indexer --file sample_conversations.json
    python -m data.indexer --file conversations.json --clear
    python -m data.indexer --stats                  # solo muestra estadísticas

El script es idempotente: llamadas repetidas hacen upsert sin duplicar.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Asegurar imports relativos correctos al ejecutar como módulo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import get_loader
from data.vector_store import clear_collection, get_stats, index_messages, index_raw_dicts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("attina.indexer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexador de conversaciones Attina → ChromaDB")
    parser.add_argument(
        "--file",
        default=None,
        help="Nombre del archivo en data/ (ej: conversations.json). Por defecto usa el dataset configurado.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Elimina la colección existente antes de indexar.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Solo muestra estadísticas de la colección actual y sale.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Tamaño de lote para indexación (default: 100).",
    )
    args = parser.parse_args()

    if args.stats:
        stats = get_stats()
        print("\n=== ChromaDB Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print()
        return

    if args.clear:
        logger.info("Limpiando colección existente...")
        clear_collection()

    loader = get_loader()

    # Resolver archivo a indexar
    if args.file:
        filename = args.file
    else:
        # Auto-detectar: preferir parquet si existe, sino JSON
        from services.config import DATA_DIR, DEFAULT_DATASET, SAMPLE_DATASET
        if (DATA_DIR / DEFAULT_DATASET).exists():
            filename = DEFAULT_DATASET
        elif (DATA_DIR / "conversations.json").exists():
            filename = "conversations.json"
        else:
            filename = SAMPLE_DATASET
            logger.warning("Dataset principal no encontrado, usando sample: %s", filename)

    logger.info("Cargando dataset: %s", filename)

    try:
        if filename.endswith(".parquet"):
            messages = loader.load_parquet(filename)
            total = index_messages(messages, batch_size=args.batch_size)
        elif filename.endswith(".json"):
            # Intentar con schema tipado primero; si falla, usar raw dicts
            try:
                messages = loader.load_json(filename)
                total = index_messages(messages, batch_size=args.batch_size)
            except Exception as schema_err:
                logger.warning("Schema tipado falló (%s), usando raw dicts.", schema_err)
                raw = loader.load_raw_json(filename)
                total = index_raw_dicts(raw, batch_size=args.batch_size)
        else:
            logger.error("Formato no soportado: %s", filename)
            sys.exit(1)

        logger.info("✅ Indexación completada. Total documentos en ChromaDB: %d", total)

    except FileNotFoundError as exc:
        logger.error("Archivo no encontrado: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
