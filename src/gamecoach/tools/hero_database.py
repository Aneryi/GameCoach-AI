from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "mock" / "heroes.json"


def get_heroes() -> dict[str, dict[str, Any]]:
    """读取全部英雄数据（内部调用）。"""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_hero(hero: str) -> dict[str, Any]:
    """读取单个英雄数据（内部调用）。"""
    heroes = get_heroes()
    return heroes[hero]


@tool
def hero_database_tool(
    hero: Optional[str] = None,
) -> dict[str, Any]:
    """查询英雄属性、定位、难度、优劣势和克制关系。

    如果不传 hero 参数则返回全部英雄列表，适合浏览英雄池。
    传入具体英雄名则返回单个英雄的详细信息。

    Args:
        hero: 英雄名称，例如 狄仁杰、孙尚香。不传则返回全部。

    Returns:
        status 为 "ok" 时 data 包含英雄信息；为 "unavailable" 时表示数据不可用。
    """
    try:
        heroes = get_heroes()
        if hero:
            if hero in heroes:
                return {"status": "ok", "hero": hero, "data": heroes[hero]}
            return {"status": "unavailable", "hero": hero, "reason": f"未找到英雄 {hero}"}
        return {"status": "ok", "heroes": list(heroes.keys()), "data": heroes}
    except FileNotFoundError:
        logger.warning("英雄数据库文件未找到: %s", DATA_PATH)
        return {"status": "unavailable", "reason": "英雄数据库暂不可用"}
    except Exception:
        logger.exception("英雄数据查询失败")
        return {"status": "unavailable", "reason": "英雄查询异常"}
