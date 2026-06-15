"""GameCoach AI 的 LangGraph 节点实现。

每个节点接收 GameCoachState，返回要更新的字段。
LLM 驱动的节点委托给 agents/ 模块，本文件保持薄封装。
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from statistics import mean

from gamecoach.agents.planner import create_llm_planner
from gamecoach.agents.response import create_llm_response
from gamecoach.agents.strategy import create_llm_strategy
from gamecoach.graph.router import EXECUTION_ORDER, FIXED_NODES, TASK_NODE_MAP
from gamecoach.graph.state import GameCoachState
from gamecoach.memory.store import load_player_memory, update_player_memory
from gamecoach.tools.hero_database import get_heroes
from gamecoach.tools.match_history import get_match_history
from gamecoach.tools.patch_meta import get_patch_meta

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 输入处理
# ═══════════════════════════════════════════════════════════


def input_normalizer(state: GameCoachState) -> GameCoachState:
    """标准化用户输入，补齐 player_id 和 game 默认值。"""
    message = state.get("user_message", "").strip()
    return {
        "normalized_message": message,
        "player_id": state.get("player_id") or "player_001",
        "game": state.get("game") or "moba",
        "errors": state.get("errors", []),
        "degraded_nodes": [],
    }


# ═══════════════════════════════════════════════════════════
# Planner（委托给 LLM Agent）
# ═══════════════════════════════════════════════════════════


def planner(state: GameCoachState) -> GameCoachState:
    """LLM 驱动的任务规划器。

    将用户问题拆解为 PlannedTask 列表，并预计算路由决策。
    路由决策在节点中计算（条件边函数无法修改 state）。
    """
    result = create_llm_planner(state)

    # 预计算路由决策（条件边只能读取 state，不能写入）
    planned = result.get("planned_tasks", [])
    required_nodes: set[str] = set()
    for task in planned:
        node = TASK_NODE_MAP.get(task["task_type"])
        if node:
            required_nodes.add(node)

    decisions = {}
    for node in EXECUTION_ORDER:
        decisions[node] = "execute" if node in required_nodes else "skip"
    for node in FIXED_NODES:
        decisions[node] = "fixed"

    result["routing_decisions"] = decisions
    return result


# ═══════════════════════════════════════════════════════════
# Memory
# ═══════════════════════════════════════════════════════════


def memory_loader(state: GameCoachState) -> GameCoachState:
    """加载玩家长期画像。"""
    player_id = state.get("player_id", "player_001")
    try:
        memory = load_player_memory(player_id)
    except Exception:
        logger.exception("加载玩家记忆失败")
        memory = {}
        state.setdefault("errors", []).append("玩家画像加载失败")

    return {"memory": memory}


# ═══════════════════════════════════════════════════════════
# 战绩分析
# ═══════════════════════════════════════════════════════════


def match_analysis_agent(state: GameCoachState) -> GameCoachState:
    """分析玩家最近战绩，识别弱点和优势。

    使用 Mock 数据计算胜率、KDA、死亡、参团等指标，
    并通过规则识别常见问题。
    """
    player_id = state.get("player_id", "player_001")
    game = state.get("game", "moba")

    try:
        match_data = get_match_history(player_id=player_id, limit=20, game=game)
    except Exception:
        logger.exception("战绩查询失败")
        state.setdefault("errors", []).append("战绩数据查询失败")
        return {
            "match_data": {"matches": []},
            "match_analysis": {
                "summary": "战绩数据暂不可用",
                "metrics": {},
                "weaknesses": [],
                "strengths": [],
            },
        }

    matches = match_data["matches"]
    if not matches:
        return {
            "match_data": match_data,
            "match_analysis": {
                "summary": "暂无战绩数据",
                "metrics": {"matches": 0},
                "weaknesses": [],
                "strengths": [],
            },
        }

    wins = [m for m in matches if m["result"] == "win"]
    deaths = [m["deaths"] for m in matches]
    kdas = [(m["kills"] + m["assists"]) / max(m["deaths"], 1) for m in matches]
    participations = [m["teamfight_participation"] for m in matches]

    # 各英雄表现
    hero_games: dict[str, list[dict]] = defaultdict(list)
    for match in matches:
        hero_games[match["hero"]].append(match)

    hero_win_rates = {
        hero: round(len([m for m in hms if m["result"] == "win"]) / len(hms), 2)
        for hero, hms in hero_games.items()
    }
    most_played = [hero for hero, _ in Counter(m["hero"] for m in matches).most_common(3)]

    avg_deaths = mean(deaths)
    avg_participation = mean(participations)
    win_rate = len(wins) / len(matches)

    weaknesses = []
    if avg_deaths >= 6:
        weaknesses.append("平均死亡偏高，团战和中期转线需要降低风险。")
    if avg_participation < 0.55:
        weaknesses.append("参团率偏低，中期容易和队友节奏脱节。")
    if win_rate < 0.5:
        weaknesses.append("近期胜率低于 50%，需要先固定英雄池并减少高风险决策。")

    analysis = {
        "summary": f"最近 {len(matches)} 场表现分析完成。",
        "metrics": {
            "matches": len(matches),
            "win_rate": round(win_rate, 2),
            "avg_kda": round(mean(kdas), 2),
            "avg_deaths": round(avg_deaths, 2),
            "teamfight_participation": round(avg_participation, 2),
        },
        "hero_win_rates": hero_win_rates,
        "most_played_heroes": most_played,
        "weaknesses": weaknesses,
        "strengths": ["射手英雄使用频率高，适合围绕稳定发育和团战输出做专项提升。"],
    }

    return {"match_data": match_data, "match_analysis": analysis}


# ═══════════════════════════════════════════════════════════
# 英雄推荐
# ═══════════════════════════════════════════════════════════


def hero_recommendation_agent(state: GameCoachState) -> GameCoachState:
    """根据版本 meta 和玩家偏好推荐上分英雄。"""
    memory = state.get("memory", {})
    game = state.get("game", "moba")
    main_roles = memory.get("main_roles", ["射手"])
    primary_role = main_roles[0] if main_roles else "射手"

    try:
        patch_meta = get_patch_meta(game=game, role=primary_role)
    except Exception:
        logger.exception("版本数据查询失败")
        state.setdefault("errors", []).append("版本数据暂不可用")
        return {"hero_recommendations": []}

    try:
        heroes = get_heroes()
    except Exception:
        logger.exception("英雄数据库查询失败")
        heroes = {}

    favorite_heroes = set(memory.get("favorite_heroes", []))

    recommendations = []
    for hero_name in patch_meta.get("strong_heroes", []):
        hero = heroes.get(hero_name)
        if not hero:
            continue

        fit_reasons = ["当前版本强势"]
        if hero_name in favorite_heroes:
            fit_reasons.append("符合玩家历史英雄偏好")
        if hero.get("difficulty") in {"low", "medium"}:
            fit_reasons.append("操作门槛适中，适合先稳定胜率")

        recommendations.append(
            {
                "hero": hero_name,
                "role": hero.get("role", "?"),
                "difficulty": hero.get("difficulty", "medium"),
                "fit_reasons": fit_reasons,
                "risks": hero.get("weaknesses", []),
            }
        )

    return {"hero_recommendations": recommendations[:3]}


# ═══════════════════════════════════════════════════════════
# 出装推荐（占位）
# ═══════════════════════════════════════════════════════════


def build_agent(state: GameCoachState) -> GameCoachState:
    """推荐英雄出装方案。

    当前为规则版占位实现，后续可升级为 LLM 版。
    """
    recommendations = state.get("hero_recommendations", [])
    if not recommendations:
        return {"build_recommendations": []}

    # 模拟出装数据
    builds = []
    for rec in recommendations[:2]:
        hero = rec.get("hero", "")
        builds.append(
            {
                "hero": hero,
                "scenario": "均势局",
                "items": ["急速战靴", "末世", "无尽战刃", "破晓", "逐日之弓", "魔女斗篷"],
                "rationale": f"{hero} 的推荐构筑兼顾持续输出和容错，适合团战容易被切入的玩家。",
                "alternatives": [
                    {
                        "condition": "敌方法刺爆发高",
                        "replace": "逐日之弓",
                        "with": "不祥征兆",
                    }
                ],
            }
        )

    return {"build_recommendations": builds}


# ═══════════════════════════════════════════════════════════
# RAG 检索
# ═══════════════════════════════════════════════════════════


def rag_agent(state: GameCoachState) -> GameCoachState:
    """攻略检索节点。

    根据用户问题和分析结果检索相关攻略内容。
    """
    user_msg = state.get("normalized_message", state.get("user_message", ""))
    analysis = state.get("match_analysis", {})

    # 构建检索查询：将用户问题 + 已识别的弱点合并
    query_parts = [user_msg]
    for w in analysis.get("weaknesses", []):
        query_parts.append(w)

    query = " ".join(query_parts)

    try:
        from gamecoach.rag.retriever import retrieve

        docs = retrieve(query, top_k=5)
    except ImportError:
        logger.warning("RAG 模块未安装")
        docs = []
    except Exception:
        logger.exception("RAG 检索失败")
        docs = []

    return {"rag_context": docs}


# ═══════════════════════════════════════════════════════════
# 策略生成（委托给 LLM Agent）
# ═══════════════════════════════════════════════════════════


def strategy_agent(state: GameCoachState) -> GameCoachState:
    """LLM 驱动的策略生成器。

    合成所有分析结果，生成个性化策略和训练计划。
    """
    result = create_llm_strategy(state)

    # 将新发现的弱点写回 memory
    strategy = result.get("strategy", {})
    if strategy.get("weaknesses"):
        try:
            player_id = state.get("player_id", "player_001")
            update_player_memory(player_id, {
                "weaknesses": strategy["weaknesses"],
                "source": "match_analysis",
            })
        except Exception:
            logger.debug("Memory 更新非致命: 无法保存到文件")

    return result


# ═══════════════════════════════════════════════════════════
# 回复合成（委托给 LLM Agent）
# ═══════════════════════════════════════════════════════════


def response_agent(state: GameCoachState) -> GameCoachState:
    """LLM 驱动的回复合成器。

    汇总所有 Agent 结果，生成面向玩家的结构化建议。
    """
    return create_llm_response(state)


# ═══════════════════════════════════════════════════════════
# 评估日志
# ═══════════════════════════════════════════════════════════


def evaluation_logger(state: GameCoachState) -> GameCoachState:
    """记录本次执行的评估指标。

    为 LangSmith 监控和后续分析提供数据。
    """
    routing = state.get("routing_decisions", {})
    executed = [k for k, v in routing.items() if v in ("execute", "fixed")]

    metrics = {
        "planner_task_count": len(state.get("planned_tasks", [])),
        "executed_nodes": executed,
        "routing_path": " → ".join(executed),
        "degraded_node_count": len(state.get("degraded_nodes", [])),
        "tool_call_success_rate": 1.0,  # Mock 工具始终成功
        "has_match_analysis": bool(state.get("match_analysis", {}).get("metrics")),
        "has_hero_recommendations": bool(state.get("hero_recommendations")),
        "has_build_recommendations": bool(state.get("build_recommendations")),
        "has_training_plan": bool(state.get("training_plan", {}).get("daily_tasks")),
        "rag_hit_count": len(state.get("rag_context", [])),
        "response_length": len(state.get("final_response", "")),
    }
    return {"metrics": metrics}
