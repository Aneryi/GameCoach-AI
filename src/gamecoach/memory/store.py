"""Player memory store.

JSON file-based persistence for player profiles.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "test_data"
CN_TZ = timezone(timedelta(hours=8))


def load_player_memory(player_id: str) -> dict[str, Any]:
    path = MEMORY_DIR / "player_memory.json"
    if not path.exists():
        logger.warning("Player memory file not found: %s", player_id)
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_player_memory(player_id: str, memory: dict[str, Any]) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / "player_memory.json"
    memory["updated_at"] = datetime.now(CN_TZ).isoformat()
    memory.setdefault("player_id", player_id)
    with path.open("w", encoding="utf-8") as file:
        json.dump(memory, file, ensure_ascii=False, indent=2)
    logger.info("Player %s memory saved", player_id)


def update_player_memory(player_id: str, updates: dict[str, Any], source: str = "inferred") -> None:
    try:
        current = load_player_memory(player_id)
    except Exception:
        current = {}

    new_weaknesses = updates.get("weaknesses", [])
    if new_weaknesses:
        existing = set(current.get("weaknesses", []))
        for w in new_weaknesses:
            existing.add(w)
        current["weaknesses"] = list(existing)

    for key in ("favorite_characters", "main_roles", "goals", "preferred_playstyle", "rank"):
        if key in updates and updates[key] is not None:
            current[key] = updates[key]

    try:
        save_player_memory(player_id, current)
    except Exception:
        logger.exception("Failed to save player %s memory", player_id)
