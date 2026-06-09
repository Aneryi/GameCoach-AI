from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "mock" / "patch_meta.json"


def get_patch_meta(game: str = "moba", role: str = "射手", patch: str = "latest") -> dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    role_meta = data[game][patch]["roles"][role]
    return {
        "game": game,
        "patch": data[game][patch]["patch"],
        "role": role,
        **role_meta,
    }

