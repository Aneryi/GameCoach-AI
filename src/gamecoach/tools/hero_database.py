"""
角色数据库工具。

提供游戏角色的属性、定位、难度、优劣势查询。
数据来源：data/test_data/characters.json。
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
    """读取全部角色数据（内部调用，不做容错）。"""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_character(name: str) -> dict[str, Any]:
    """读取单个角色数据（内部调用，不做容错，调用方负责处理 KeyError）。"""
    characters = get_characters()
    return characters[name]


@tool
def character_database_tool(
    name: Optional[str] = None,
) -> dict[str, Any]:
    """
    查询游戏角色属性、定位、难度和优劣势。

    不传 name 参数返回全部角色列表，传入具体名称返回单个角色的详细信息。
    适合浏览角色池或查询特定角色的克制关系。

    Args:
        name: 角色名称，例如 Alpha、Bravo。不传则返回全部角色列表。

    Returns:
        status="ok" 时 data 包含角色信息；status="unavailable" 时表示数据不可用。
    """
    try:
        characters = get_characters()
        if name:
            if name in characters:
                return {"status": "ok", "name": name, "data": characters[name]}
            return {"status": "unavailable", "reason": f"未找到角色 {name}"}
        return {"status": "ok", "characters": list(characters.keys()), "data": characters}
    except FileNotFoundError:
        logger.warning("角色数据库文件未找到: %s", DATA_PATH)
        return {"status": "unavailable", "reason": "角色数据库暂不可用"}
    except Exception:
        logger.exception("角色查询失败")
        return {"status": "unavailable", "reason": "角色查询异常"}
