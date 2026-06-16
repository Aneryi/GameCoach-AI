"""Meta / patch data tool.

查询当前版本的强势角色、削弱角色和装备改动信息。
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
    return {
        "patch": data["game"][patch]["patch"],
        "role": role,
        **role_meta,
    }


@tool
def patch_meta_tool(
    role: Optional[str] = None,
    patch: str = "latest",
) -> dict[str, Any]:
    """查询当前版本的强势角色和装备改动。

    用于结合版本趋势做角色推荐和出装建议。
    可指定角色位置过滤，不传则返回 damage 位置。

    Args:
        role: 角色位置过滤，如 damage、tank、support。不传返回 damage。
        patch: 版本号，默认 latest。

    Returns:
        status 为 "ok" 时包含版本号、强势角色列表、装备改动等。
    """
    try:
        result = get_patch_meta(role=role or "damage", patch=patch)
        return {"status": "ok", **result}
    except FileNotFoundError:
        logger.warning("Patch data file not found: %s", DATA_PATH)
        return {"status": "unavailable", "reason": "Patch data unavailable"}
    except KeyError:
        return {"status": "unavailable", "reason": f"No patch data for {patch}/{role}"}
    except Exception:
        logger.exception("Patch meta query failed")
        return {"status": "unavailable", "reason": "Query error"}
