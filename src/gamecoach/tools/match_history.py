"""
战绩查询工具。

提供玩家近期对战记录的查询接口。数据来源：data/test_data/matches.json。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# langchain_core.tools.tool: LangChain @tool 装饰器，将函数注册为可被 LLM 调用的工具
# 带 @tool 的函数自动生成 name/description schema，供 LLM tool calling 使用
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Path(__file__).resolve().parents[3] 上溯到项目根目录
# 从 tools/match_history.py → tools → gamecoach → src → 项目根
DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "test_data" / "matches.json"


def get_match_history(player_id: str, limit: int = 20) -> dict[str, Any]:
    """读取 Mock 对战数据（内部调用，不做容错，由 @tool 版本负责容错）。"""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    matches = data.get(player_id, data["player_001"])[:limit]
    return {"player_id": player_id, "limit": limit, "matches": matches}


@tool
def match_history_tool(
    player_id: str,
    limit: int = 20,
) -> dict[str, Any]:
    """
    查询玩家最近 N 场对局战绩。

    返回每局的胜负、角色、KDA、评分、伤害、参团率等数据。
    用于分析玩家近期表现和定位问题。

    Args:
        player_id: 玩家 ID，例如 player_001。
        limit: 查询场数，默认 20。

    Returns:
        包含 status ("ok"/"unavailable")、matches 列表的字典。
        当数据文件缺失或解析失败时返回 status="unavailable"。
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
        logger.warning("战绩数据文件未找到: %s", DATA_PATH)
        return {"status": "unavailable", "matches": [], "reason": "战绩数据暂不可用"}
    except Exception:
        logger.exception("战绩查询失败")
        return {"status": "unavailable", "matches": [], "reason": "战绩查询异常"}
