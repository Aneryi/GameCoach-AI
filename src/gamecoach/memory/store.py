"""玩家记忆存储。

提供玩家长期画像的读取、写入和合并更新。
MVP 阶段使用 JSON 文件存储，生产环境应切换到数据库。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"

# 北京时间 (UTC+8)
CN_TZ = timezone(timedelta(hours=8))


def load_player_memory(player_id: str) -> dict[str, Any]:
    """读取玩家长期画像。

    Args:
        player_id: 玩家 ID。

    Returns:
        玩家画像字典，如果文件不存在则回退到 player_001。
    """
    path = MEMORY_DIR / f"{player_id}.json"
    if not path.exists():
        path = MEMORY_DIR / "player_001.json"
    if not path.exists():
        logger.warning("找不到玩家记忆文件: %s", player_id)
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_player_memory(player_id: str, memory: dict[str, Any]) -> None:
    """持久化玩家画像到 JSON 文件。

    Args:
        player_id: 玩家 ID。
        memory: 完整玩家画像字典。
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / f"{player_id}.json"

    memory["updated_at"] = datetime.now(CN_TZ).isoformat()
    memory.setdefault("player_id", player_id)

    with path.open("w", encoding="utf-8") as file:
        json.dump(memory, file, ensure_ascii=False, indent=2)

    logger.info("玩家 %s 的记忆已保存", player_id)


def update_player_memory(
    player_id: str,
    updates: dict[str, Any],
    source: str = "inferred",
) -> None:
    """增量更新玩家画像，合并新旧数据。

    两层记忆模型：
    - 来自 source="match_analysis" 的更新存入 confirmed（高置信度）
    - 来自 source="inferred" 的更新存入 pending（待确认）

    Args:
        player_id: 玩家 ID。
        updates: 要合并的字段。
        source: 更新来源 — "match_analysis" 或 "inferred"。
    """
    try:
        current = load_player_memory(player_id)
    except Exception:
        current = {}

    if source == "match_analysis":
        target_key = "confirmed"
    else:
        target_key = "pending"

    if target_key not in current:
        current[target_key] = {}

    # 合并弱点（去重）
    new_weaknesses = updates.get("weaknesses", [])
    if new_weaknesses:
        existing = set(current.get("weaknesses", []))
        for w in new_weaknesses:
            existing.add(w)
        current["weaknesses"] = list(existing)

    # 合并其他字段
    for key in ("favorite_heroes", "main_roles", "goals", "preferred_playstyle", "rank"):
        if key in updates and updates[key] is not None:
            current[key] = updates[key]

    try:
        save_player_memory(player_id, current)
    except Exception:
        logger.exception("保存玩家 %s 的记忆失败", player_id)
