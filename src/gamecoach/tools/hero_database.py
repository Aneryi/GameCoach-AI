from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "mock" / "heroes.json"


def get_heroes() -> dict[str, dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_hero(hero: str) -> dict[str, Any]:
    heroes = get_heroes()
    return heroes[hero]

