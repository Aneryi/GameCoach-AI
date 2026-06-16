"""Character database tool.

查询游戏角色属性、定位、难度、优劣势等数据。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "test_data" / "characters.json"


def get_characters() -> dict[str, dict[str, Any]]:
    """读取全部角色数据（内部调用）。"""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_character(name: str) -> dict[str, Any]:
    """读取单个角色数据（内部调用）。"""
    characters = get_characters()
    return characters[name]


@tool
def character_database_tool(
    name: Optional[str] = None,
) -> dict[str, Any]:
    """查询游戏角色属性、定位、难度和优劣势。

    不传 name 返回全部角色列表，传入具体名称返回详细信息。

    Args:
        name: 角色名称，例如 Alpha、Bravo。不传则返回全部。

    Returns:
        status 为 "ok" 时 data 包含角色信息。
    """
    try:
        characters = get_characters()
        if name:
            if name in characters:
                return {"status": "ok", "name": name, "data": characters[name]}
            return {"status": "unavailable", "reason": f"Character {name} not found"}
        return {"status": "ok", "characters": list(characters.keys()), "data": characters}
    except FileNotFoundError:
        logger.warning("Character database file not found: %s", DATA_PATH)
        return {"status": "unavailable", "reason": "Character database unavailable"}
    except Exception:
        logger.exception("Character query failed")
        return {"status": "unavailable", "reason": "Query error"}
