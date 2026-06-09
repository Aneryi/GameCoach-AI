from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "mock" / "matches.json"


def get_match_history(player_id: str, limit: int = 20, game: str = "moba") -> dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    matches = data.get(player_id, data["player_001"])[:limit]
    return {"player_id": player_id, "game": game, "limit": limit, "matches": matches}

