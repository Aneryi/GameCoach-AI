"""
玩家记忆存储（JSON 文件持久化）。

提供玩家长期画像的读取、写入和增量更新。
MVP 阶段使用 JSON 文件存储，生产环境应切换到数据库。

两层记忆模型（简化版）：
- 来自 match_analysis 数据源的更新：高置信度，直接合并
- 来自 inferred 推断的更新：低置信度，仅追加到 pending
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "test_data"

# 北京时间 (UTC+8) —— updated_at 时间戳使用
CN_TZ = timezone(timedelta(hours=8))


def load_player_memory(player_id: str) -> dict[str, Any]:
    """
    读取玩家长期画像。

    从 data/test_data/player_memory.json 加载。
    如果文件不存在，返回空 dict（不阻断后续节点）。

    Args:
        player_id: 玩家 ID（当前 MVP 阶段统一使用 player_memory.json）。

    Returns:
        玩家画像字典，含 favorite_characters / main_roles / weaknesses / goals 等。
    """
    path = MEMORY_DIR / "player_memory.json"
    if not path.exists():
        logger.warning("玩家记忆文件未找到: %s", player_id)
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_player_memory(player_id: str, memory: dict[str, Any]) -> None:
    """
    持久化玩家画像到 JSON 文件。

    自动添加 updated_at 时间戳和 player_id。
    目录不存在时自动创建。

    Args:
        player_id: 玩家 ID。
        memory: 完整玩家画像字典。
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / "player_memory.json"
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
    """
    增量更新玩家画像（合并而非覆盖）。

    合并逻辑：
    - weaknesses: 去重追加（不覆盖已有弱点）
    - favorite_characters / main_roles / goals / rank: 新值覆盖旧值

    Args:
        player_id: 玩家 ID。
        updates: 要合并的字段。
        source: 更新来源 — "match_analysis"（高置信）或 "inferred"（低置信）。
    """
    try:
        current = load_player_memory(player_id)
    except Exception:
        current = {}

    # 弱点去重追加
    new_weaknesses = updates.get("weaknesses", [])
    if new_weaknesses:
        existing = set(current.get("weaknesses", []))
        for w in new_weaknesses:
            existing.add(w)
        current["weaknesses"] = list(existing)

    # 其他字段覆盖
    for key in ("favorite_characters", "main_roles", "goals", "preferred_playstyle", "rank"):
        if key in updates and updates[key] is not None:
            current[key] = updates[key]

    try:
        save_player_memory(player_id, current)
    except Exception:
        logger.exception("保存玩家 %s 的记忆失败", player_id)
