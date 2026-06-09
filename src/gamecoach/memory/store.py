from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"


def load_player_memory(player_id: str) -> dict[str, Any]:
    path = MEMORY_DIR / f"{player_id}.json"
    if not path.exists():
        path = MEMORY_DIR / "player_001.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

