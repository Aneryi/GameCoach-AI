"""
版本信息工具。

查询当前游戏版本的 meta 数据：强势角色、削弱角色、装备改动。
用于角色推荐和出装建议的版本依据。
数据来源：data/test_data/patch_meta.json。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "test_data" / "patch_meta.json"


def get_patch_meta(role: str = "damage", patch: str = "latest") -> dict[str, Any]:
    """读取版本元数据（内部调用）。"""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    role_meta = data["game"][patch]["roles"][role]
    return {"patch": data["game"][patch]["patch"], "role": role, **role_meta}


@tool
def patch_meta_tool(
    role: Optional[str] = None,
    patch: str = "latest",
) -> dict[str, Any]:
    """
    查询当前版本的强势角色和装备改动。

    用于结合版本趋势做角色推荐和出装建议。
    可指定角色位置过滤。

    Args:
        role: 角色位置过滤，如 damage、tank、support。不传默认 damage。
        patch: 版本号，默认 latest。

    Returns:
        status="ok" 时包含 strong_characters、nerfed_characters、buffed_items 等。
    """
    try:
        result = get_patch_meta(role=role or "damage", patch=patch)
        return {"status": "ok", **result}
    except FileNotFoundError:
        logger.warning("版本数据文件未找到: %s", DATA_PATH)
        return {"status": "unavailable", "reason": "版本数据暂不可用"}
    except KeyError:
        return {"status": "unavailable", "reason": f"未找到 {patch}/{role} 的版本数据"}
    except Exception:
        logger.exception("版本数据查询失败")
        return {"status": "unavailable", "reason": "版本查询异常"}
