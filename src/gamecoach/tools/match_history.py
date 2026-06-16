"""Match history tool.

查询玩家最近对战记录，返回胜负、角色、KDA、评分等数据。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "test_data" / "matches.json"


def get_match_history(player_id: str, limit: int = 20) -> dict[str, Any]:
    """读取比赛记录（内部调用）。"""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    matches = data.get(player_id, data["player_001"])[:limit]
    return {"player_id": player_id, "limit": limit, "matches": matches}


@tool
def match_history_tool(
    player_id: str,
    limit: int = 20,
) -> dict[str, Any]:
    """查询玩家最近对战记录。

    返回每局的胜负、角色、KDA、评分、伤害、参团率等数据，
    用于分析玩家近期表现和定位问题。

    Args:
        player_id: 玩家 ID，例如 player_001。
        limit: 查询场数，默认 20。

    Returns:
        包含 status、matches 列表的字典。
    """
    try:
        data = get_match_history(player_id=player_id, limit=limit)
        matches = data["matches"]
        return {
            "status": "ok",
            "player_id": player_id,
            "limit": limit,
            "total_matches": len(matches),
            "matches": matches,
        }
    except FileNotFoundError:
        logger.warning("Match data file not found: %s", DATA_PATH)
        return {"status": "unavailable", "matches": [], "reason": "Match data unavailable"}
    except Exception:
        logger.exception("Match history query failed")
        return {"status": "unavailable", "matches": [], "reason": "Query error"}
