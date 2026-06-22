"""
装备/出装工具。

提供装备数据库查询和按场景（顺风/均势/逆风）的出装推荐。
数据来源：data/test_data/items.json。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "test_data" / "items.json"


def get_all_items() -> dict[str, dict[str, Any]]:
    """读取全部装备数据。"""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@tool
def build_tool(
    character: Optional[str] = None,
    role: Optional[str] = None,
    scenario: str = "balanced",
) -> dict[str, Any]:
    """
    查询推荐出装方案。

    根据角色、位置和对局形势（顺风/均势/逆风）推荐装备搭配。

    Args:
        character: 角色名称，如 Alpha、Bravo。不传按位置推荐通用方案。
        role: 位置过滤，如 damage、tank、support。
        scenario: 对局形势 — ahead(顺风) / balanced(均势) / behind(逆风)，默认 balanced。

    Returns:
        status="ok" 时包含 recommended 装备列表。
    """
    try:
        all_items = get_all_items()
        if role:
            available = {
                k: v for k, v in all_items.items()
                if "all" in v.get("role_fit", []) or role in v.get("role_fit", [])
            }
        else:
            available = all_items

        # 按装备类别分组
        boots = [{"name": k, **v} for k, v in available.items() if v["category"] == "boots"]
        attack = [{"name": k, **v} for k, v in available.items() if v["category"] == "attack"]
        defense = [{"name": k, **v} for k, v in available.items() if v["category"] == "defense"]

        # 根据对局形势选择装备（规则版）
        if scenario == "ahead":
            rec = [b for b in boots if b["name"] == "Basic_Boots"]
            rec += [a for a in attack if a["name"] in ("Core_Item", "Crit_Core", "Burst_Item", "Pen_Item")]
        elif scenario == "behind":
            rec = [b for b in boots if b["name"] == "Tenacity_Boots"]
            rec += [a for a in attack if a["name"] in ("Core_Item", "Pen_Item")]
            rec += [d for d in defense if d["name"] in ("Armor_Item", "Magic_Resist")]
        else:  # balanced
            rec = [b for b in boots if b["name"] == "Basic_Boots"]
            rec += [a for a in attack if a["name"] in ("Core_Item", "Crit_Core", "Pen_Item", "Range_Bow")]
            rec += [d for d in defense if d["name"] == "Magic_Resist"]

        return {
            "status": "ok",
            "character": character, "role": role, "scenario": scenario,
            "recommended": rec,
        }
    except FileNotFoundError:
        logger.warning("装备数据库文件未找到")
        return {"status": "unavailable", "reason": "装备数据库暂不可用"}
    except Exception:
        logger.exception("出装查询失败")
        return {"status": "unavailable", "reason": "出装查询异常"}
