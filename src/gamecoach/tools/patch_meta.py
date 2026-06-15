from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "mock" / "patch_meta.json"


def get_patch_meta(
    game: str = "moba", role: str = "射手", patch: str = "latest"
) -> dict[str, Any]:
    """读取版本元数据（内部调用）。"""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    role_meta = data[game][patch]["roles"][role]
    return {
        "game": game,
        "patch": data[game][patch]["patch"],
        "role": role,
        **role_meta,
    }


@tool
def patch_meta_tool(
    game: str = "moba",
    role: Optional[str] = None,
    patch: str = "latest",
) -> dict[str, Any]:
    """查询当前游戏版本的强势英雄、削弱英雄和装备改动。

    用于结合版本趋势做英雄推荐和出装建议。
    可指定位置（role）过滤，不传则返回全部位置。

    Args:
        game: 游戏类型，默认 moba。
        role: 位置过滤，例如 射手、打野、中单。不传返回全部。
        patch: 版本号，默认 latest。

    Returns:
        status 为 "ok" 时包含版本号、强势英雄列表、装备改动等。
    """
    try:
        result = get_patch_meta(game=game, role=role or "射手", patch=patch)
        return {"status": "ok", **result}
    except FileNotFoundError:
        logger.warning("版本数据文件未找到: %s", DATA_PATH)
        return {"status": "unavailable", "reason": "版本数据暂不可用"}
    except KeyError:
        return {"status": "unavailable", "reason": f"未找到 {game}/{patch}/{role} 的版本数据"}
    except Exception:
        logger.exception("版本数据查询失败")
        return {"status": "unavailable", "reason": "版本查询异常"}
