import json
import pandas as pd
from pathlib import Path
from typing import Generator
from .schema import Message


class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._cache: dict[str, list[Message]] = {}

    def load_json(self, filename: str) -> list[Message]:
        if filename in self._cache:
            return self._cache[filename]

        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = [Message.from_dict(item) for item in data]
        self._cache[filename] = messages
        return messages

    def load_raw_json(self, filename: str) -> list[dict]:
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(self, filename: str, data: list[dict]) -> None:
        filepath = self.data_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def stream_messages(self, filename: str, batch_size: int = 100) -> Generator[list[Message], None, None]:
        messages = self.load_json(filename)
        for i in range(0, len(messages), batch_size):
            yield messages[i:i + batch_size]

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_stats(self, filename: str) -> dict:
        messages = self.load_json(filename)
        return {
            "total_messages": len(messages),
            "total_authors": len(set(m.author.id for m in messages)),
            "total_threads": len(set(m.thread_id for m in messages if m.thread_id)),
            "date_range": {
                "min": min(m.created_at for m in messages).isoformat(),
                "max": max(m.created_at for m in messages).isoformat(),
            },
        }


_loader = DataLoader()


def get_loader() -> DataLoader:
    return _loader