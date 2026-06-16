"""Equipment / build tool."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "test_data" / "items.json"


def get_all_items() -> dict[str, dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@tool
def build_tool(
    character: Optional[str] = None,
    role: Optional[str] = None,
    scenario: str = "balanced",
) -> dict[str, Any]:
    """Query recommended equipment builds.

    Args:
        character: Character name, e.g. Alpha, Bravo.
        role: Role filter, e.g. damage, tank, support.
        scenario: Game state — ahead / balanced / behind. Default balanced.

    Returns:
        Build recommendations with rationale.
    """
    try:
        all_items = get_all_items()
        if role:
            available = {k: v for k, v in all_items.items() if "all" in v.get("role_fit", []) or role in v.get("role_fit", [])}
        else:
            available = all_items

        boots = [{"name": k, **v} for k, v in available.items() if v["category"] == "boots"]
        attack = [{"name": k, **v} for k, v in available.items() if v["category"] == "attack"]
        defense = [{"name": k, **v} for k, v in available.items() if v["category"] == "defense"]

        if scenario == "ahead":
            rec = [b for b in boots if b["name"] == "Basic_Boots"]
            rec += [a for a in attack if a["name"] in ("Core_Item", "Crit_Core", "Burst_Item", "Pen_Item")]
        elif scenario == "behind":
            rec = [b for b in boots if b["name"] == "Tenacity_Boots"]
            rec += [a for a in attack if a["name"] in ("Core_Item", "Pen_Item")]
            rec += [d for d in defense if d["name"] in ("Armor_Item", "Magic_Resist")]
        else:
            rec = [b for b in boots if b["name"] == "Basic_Boots"]
            rec += [a for a in attack if a["name"] in ("Core_Item", "Crit_Core", "Pen_Item", "Range_Bow")]
            rec += [d for d in defense if d["name"] == "Magic_Resist"]

        return {"status": "ok", "character": character, "role": role, "scenario": scenario, "recommended": rec}
    except FileNotFoundError:
        logger.warning("Items data file not found")
        return {"status": "unavailable", "reason": "Item database unavailable"}
    except Exception:
        logger.exception("Build query failed")
        return {"status": "unavailable", "reason": "Query error"}
