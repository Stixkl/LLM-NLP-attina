import json
import os
from functools import lru_cache
from typing import Optional

from data.schema import Message

# Parquet reading requires pyarrow or fastparquet.
# If neither is available we fall back to the bundled sample JSON.
try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False


_PARQUET_ENGINES = ["pyarrow", "fastparquet"]


def _try_read_parquet(path: str):
    """Try to read a parquet file using any available engine."""
    if not _PANDAS_OK:
        return None
    import pandas as pd
    for engine in _PARQUET_ENGINES:
        try:
            return pd.read_parquet(path, engine=engine)
        except Exception:
            continue
    return None


def _df_to_messages(df) -> list[Message]:
    """Convert a DataFrame row by row into Message objects."""
    messages = []
    for _, row in df.iterrows():
        data = {}
        for field in Message.model_fields:
            if field in row.index:
                val = row[field]
                # pandas NaN → None
                try:
                    import math
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        val = None
                except Exception:
                    pass
                data[field] = val
        try:
            messages.append(Message(**data))
        except Exception:
            pass
    return messages


def _json_to_messages(records: list[dict]) -> list[Message]:
    """Convert a list of dicts (from JSON) into Message objects."""
    messages = []
    for record in records:
        try:
            messages.append(Message(**record))
        except Exception:
            pass
    return messages


class DataLoader:
    """
    Loads conversation data from a parquet or JSON file and
    exposes a list of Message objects with caching.
    """

    def __init__(self, path: str):
        self.path = path
        self._messages: Optional[list[Message]] = None

    def load(self) -> list[Message]:
        if self._messages is not None:
            return self._messages

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        ext = os.path.splitext(self.path)[1].lower()

        if ext == ".parquet":
            df = _try_read_parquet(self.path)
            if df is not None:
                self._messages = _df_to_messages(df)
            else:
                raise RuntimeError(
                    "Cannot read parquet: install pyarrow or fastparquet.\n"
                    "Run: pip install pyarrow"
                )
        elif ext == ".json":
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            records = raw if isinstance(raw, list) else raw.get("data", [raw])
            self._messages = _json_to_messages(records)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return self._messages

    def __len__(self) -> int:
        return len(self.load())


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_DEFAULT_PATHS = [
    "data/Reto_data_20251023_122206.parquet",
    "data/sample_conversations.json",
]


@lru_cache(maxsize=1)
def get_loader() -> DataLoader:
    """Return a cached DataLoader pointing to the first available dataset."""
    for path in _DEFAULT_PATHS:
        if os.path.exists(path):
            return DataLoader(path)
    raise FileNotFoundError(
        "No dataset found. Place the parquet file or sample JSON in the data/ folder."
    )


def load_messages() -> list[Message]:
    """Convenience function: load and return all messages."""
    return get_loader().load()