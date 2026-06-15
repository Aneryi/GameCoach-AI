from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "mock" / "matches.json"


def get_match_history(player_id: str, limit: int = 20, game: str = "moba") -> dict[str, Any]:
    """读取 Mock 战绩数据（内部调用，不做容错）。"""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    matches = data.get(player_id, data["player_001"])[:limit]
    return {"player_id": player_id, "game": game, "limit": limit, "matches": matches}


@tool
def match_history_tool(
    player_id: str,
    limit: int = 20,
    game: str = "moba",
) -> dict[str, Any]:
    """查询玩家最近 N 场对局战绩。

    返回每局的胜负、英雄、KDA、经济、伤害、参团率等数据。
    用于分析玩家近期表现和定位问题。

    Args:
        player_id: 玩家 ID，例如 player_001。
        limit: 查询场数，默认 20。
        game: 游戏类型，默认 moba。

    Returns:
        包含 status、matches 列表和统计摘要的字典。
        status 为 "ok" 时正常，为 "unavailable" 时表示数据不可用。
    """
    try:
        data = get_match_history(player_id=player_id, limit=limit, game=game)
        matches = data["matches"]
        return {
            "status": "ok",
            "player_id": player_id,
            "game": game,
            "limit": limit,
            "total_matches": len(matches),
            "matches": matches,
        }
    except FileNotFoundError:
        logger.warning("战绩数据文件未找到: %s", DATA_PATH)
        return {"status": "unavailable", "matches": [], "reason": "战绩数据暂不可用"}
    except Exception:
        logger.exception("战绩查询失败")
        return {"status": "unavailable", "matches": [], "reason": "战绩查询异常"}
