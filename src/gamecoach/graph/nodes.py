"""
GameCoach AI 的 LangGraph 节点实现。

每个节点是一个纯函数，接收 GameCoachState，返回要更新的字段 dict。
LangGraph 自动将返回的 dict 合并到 state（partial update）。

节点分类：
- 输入处理: input_normalizer — 标准化用户输入
- 任务规划: planner — LLM 拆解任务 + 预计算路由决策
- 数据加载: memory_loader — 读取玩家长期画像
- 数据分析: match_analysis_agent — 战绩指标计算与短板诊断
- 推荐层:   character_recommendation_agent — 版本 meta + 玩家偏好 → 角色推荐
           build_agent — 角色 + 装备库 + LLM → 出装推荐
           rag_agent — FAISS 向量检索 → 攻略片段
- 策略合成: strategy_agent — LLM 综合所有数据生成策略和训练计划
- 回复生成: response_agent — LLM 汇总所有结果生成玩家友好的回复
- 评估日志: evaluation_logger — 记录执行指标到 state.metrics
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from statistics import mean

# Agent 工厂函数：接收 state → 调用 LLM → 返回 state 更新
from gamecoach.agents.planner import create_llm_planner
from gamecoach.agents.response import create_llm_response
from gamecoach.agents.strategy import create_llm_strategy

# LLM 模型工厂：用于 build_agent 中直接调用 LLM（不走 Agent 封装）
from gamecoach.config.llm import get_chat_model

# 路由常量和辅助函数：用于在 planner 节点中预计算路由决策
from gamecoach.graph.router import EXECUTION_ORDER, FIXED_NODES, TASK_NODE_MAP
from gamecoach.graph.state import GameCoachState

# Memory: JSON 文件读写
from gamecoach.memory.store import load_player_memory, update_player_memory

# 工具函数：内部调用（非 @tool 版本，不需要 LLM 上下文即可直接调用）
from gamecoach.tools.hero_database import get_characters
from gamecoach.tools.match_history import get_match_history
from gamecoach.tools.patch_meta import get_patch_meta

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 输入处理
# ═══════════════════════════════════════════════════════════════

def input_normalizer(state: GameCoachState) -> GameCoachState:
    """
    标准化用户输入。

    做的事情：
    1. 去除首尾空格
    2. 补全默认 player_id（"player_001"）
    3. 初始化 errors 和 degraded_nodes 为空列表

    这是 DAG 的第一个节点，确保后续节点收到的 state 字段完整。
    """
    message = state.get("user_message", "").strip()
    return {
        "normalized_message": message,
        "player_id": state.get("player_id") or "player_001",
        "errors": state.get("errors", []),
        "degraded_nodes": [],
    }


# ═══════════════════════════════════════════════════════════════
# Planner
# ═══════════════════════════════════════════════════════════════

def planner(state: GameCoachState) -> GameCoachState:
    """
    LLM 驱动的任务规划器 + 路由决策预计算。

    两步操作：
    1. 调用 create_llm_planner 生成 planned_tasks（LLM 或 fallback）
    2. 预计算 routing_decisions —— 在节点中而非条件边函数中计算，
       因为 LangGraph 的条件边函数无法修改 state（传入的是快照副本）。

    routing_decisions 格式：
        {"memory_loader": "execute", "match_analysis_agent": "skip", ...}
    其中 "execute"=需要执行, "skip"=跳过, "fixed"=固定执行（不受路由影响）。
    """
    result = create_llm_planner(state)

    # 从 planned_tasks 提取需要的节点集合
    planned = result.get("planned_tasks", [])
    required_nodes: set[str] = set()
    for task in planned:
        node = TASK_NODE_MAP.get(task["task_type"])
        if node:
            required_nodes.add(node)

    # 按 EXECUTION_ORDER 顺序标记每个节点是执行还是跳过
    decisions = {}
    for node in EXECUTION_ORDER:
        decisions[node] = "execute" if node in required_nodes else "skip"
    for node in FIXED_NODES:
        decisions[node] = "fixed"

    result["routing_decisions"] = decisions
    return result


# ═══════════════════════════════════════════════════════════════
# Memory
# ═══════════════════════════════════════════════════════════════

def memory_loader(state: GameCoachState) -> GameCoachState:
    """
    加载玩家长期画像。

    从 data/test_data/player_memory.json 读取玩家偏好、弱点、目标等。
    如果文件不存在或解析失败，返回空 memory 并记录错误（不阻断流程）。
    """
    player_id = state.get("player_id", "player_001")
    try:
        memory = load_player_memory(player_id)
    except Exception:
        logger.exception("Memory load failed")
        memory = {}
        state.setdefault("errors", []).append("Player memory load failed")
    return {"memory": memory}


# ═══════════════════════════════════════════════════════════════
# 战绩分析
# ═══════════════════════════════════════════════════════════════

def match_analysis_agent(state: GameCoachState) -> GameCoachState:
    """
    战绩分析代理。

    从 Mock 数据（data/test_data/matches.json）读取最近 20 场对局，
    计算核心指标并通过规则识别玩家短板。

    计算的指标：
    - win_rate: 胜率 = 胜场 / 总场次
    - avg_kda: 平均 KDA = (击杀 + 助攻) / max(死亡, 1)，处理零死亡情况
    - avg_deaths: 平均死亡次数
    - teamfight_participation: 平均参团率
    - character_win_rates: 每个角色的胜率
    - most_played_characters: 最常使用的角色 Top 3

    弱点诊断（规则）：
    - avg_deaths >= 6 → 死亡偏高
    - participation < 0.55 → 参团率低
    - win_rate < 0.5 → 胜率低

    异常处理：数据文件缺失时返回空分析，不阻断后续节点。
    """
    player_id = state.get("player_id", "player_001")
    try:
        match_data = get_match_history(player_id=player_id, limit=20)
    except Exception:
        logger.exception("Match query failed")
        return {
            "match_data": {"matches": []},
            "match_analysis": {
                "summary": "Data unavailable", "metrics": {},
                "weaknesses": [], "strengths": [],
            },
        }

    matches = match_data["matches"]
    if not matches:
        return {
            "match_data": match_data,
            "match_analysis": {
                "summary": "No match data", "metrics": {"matches": 0},
                "weaknesses": [], "strengths": [],
            },
        }

    # 统计计算
    wins = [m for m in matches if m["result"] == "win"]
    deaths = [m["deaths"] for m in matches]
    # KDA 计算：max(deaths, 1) 避免除零
    kdas = [(m["kills"] + m["assists"]) / max(m["deaths"], 1) for m in matches]
    participations = [m["teamfight_participation"] for m in matches]

    # 按角色分组统计胜率
    char_games: dict[str, list[dict]] = defaultdict(list)
    for match in matches:
        char_games[match["character"]].append(match)

    char_win_rates = {
        char: round(len([m for m in cms if m["result"] == "win"]) / len(cms), 2)
        for char, cms in char_games.items()
    }
    # Counter.most_common(3): 统计最常使用的 3 个角色
    most_played = [char for char, _ in Counter(
        m["character"] for m in matches
    ).most_common(3)]

    avg_deaths = mean(deaths)
    avg_participation = mean(participations)
    win_rate = len(wins) / len(matches)

    # 规则诊断弱点
    weaknesses = []
    if avg_deaths >= 6:
        weaknesses.append("Average deaths too high – reduce risky positioning.")
    if avg_participation < 0.55:
        weaknesses.append("Teamfight participation low – improve mid-game rotation timing.")
    if win_rate < 0.5:
        weaknesses.append("Win rate below 50% – narrow character pool and reduce high-risk plays.")

    analysis = {
        "summary": f"Analysis of {len(matches)} recent matches complete.",
        "metrics": {
            "matches": len(matches),
            "win_rate": round(win_rate, 2),
            "avg_kda": round(mean(kdas), 2),
            "avg_deaths": round(avg_deaths, 2),
            "teamfight_participation": round(avg_participation, 2),
        },
        "character_win_rates": char_win_rates,
        "most_played_characters": most_played,
        "weaknesses": weaknesses,
        "strengths": [
            "Consistent main-role usage provides a solid foundation for targeted improvement."
        ],
    }
    return {"match_data": match_data, "match_analysis": analysis}


# ═══════════════════════════════════════════════════════════════
# 角色推荐
# ═══════════════════════════════════════════════════════════════

def character_recommendation_agent(state: GameCoachState) -> GameCoachState:
    """
    角色推荐代理。

    结合三方面信息推荐上分角色：
    1. 玩家主玩位置（从 memory.main_roles 读取）
    2. 当前版本强势角色（从 patch_meta 工具获取）
    3. 玩家历史偏好角色（从 memory.favorite_characters 读取）

    推荐理由（fit_reasons）：
    - "Strong in current meta": 版本强势
    - "Matches player character preference": 玩家历史偏好
    - "Manageable difficulty for consistent results": 操作门槛低

    返回前 3 个推荐角色。
    """
    memory = state.get("memory", {})
    main_roles = memory.get("main_roles", ["damage"])
    primary_role = main_roles[0] if main_roles else "damage"

    try:
        patch_meta = get_patch_meta(role=primary_role)
    except Exception:
        logger.exception("Patch data query failed")
        return {"character_recommendations": []}

    try:
        characters = get_characters()
    except Exception:
        logger.exception("Character database query failed")
        characters = {}

    favorite_characters = set(memory.get("favorite_characters", []))

    recommendations = []
    for char_name in patch_meta.get("strong_characters", []):
        char = characters.get(char_name)
        if not char:
            continue
        fit_reasons = ["Strong in current meta"]
        if char_name in favorite_characters:
            fit_reasons.append("Matches player character preference")
        if char.get("difficulty") in {"low", "medium"}:
            fit_reasons.append("Manageable difficulty for consistent results")
        recommendations.append({
            "character": char_name,
            "role": char.get("role", "?"),
            "difficulty": char.get("difficulty", "medium"),
            "fit_reasons": fit_reasons,
            "risks": char.get("weaknesses", []),
        })

    return {"character_recommendations": recommendations[:3]}


# ═══════════════════════════════════════════════════════════════
# 出装推荐
# ═══════════════════════════════════════════════════════════════

def build_agent(state: GameCoachState) -> GameCoachState:
    """
    出装推荐代理。

    根据推荐的角色和玩家位置，结合装备数据库生成出装方案。
    优先使用 LLM 生成个性化推荐，LLM 不可用时用规则版 fallback。

    LLM 输出格式：
        Build: Item1 -> Item2 -> ...
        Rationale: 推荐理由
        Scenario: balanced / ahead / behind
    """
    recommendations = state.get("character_recommendations", [])
    memory = state.get("memory", {})
    main_roles = memory.get("main_roles", ["damage"])
    role = main_roles[0] if main_roles else "damage"

    try:
        from gamecoach.tools.items import get_all_items
        all_items = get_all_items()
    except Exception:
        logger.exception("Item data query failed")
        return {"build_recommendations": []}

    llm = get_chat_model(temperature=0.2)
    char_names = [r.get("character", "") for r in recommendations[:2]] if recommendations else ["any"]

    # Fallback: LLM 不可用时的规则推荐
    if llm is None:
        boots = [{"name": k, **v} for k, v in all_items.items() if v["category"] == "boots"][:1]
        attack = [{"name": k, **v} for k, v in all_items.items() if v["category"] == "attack"][:4]
        defense = [{"name": k, **v} for k, v in all_items.items() if v["category"] == "defense"][:1]
        fallback = [i["name"] for i in boots + attack + defense]
        return {"build_recommendations": [{
            "character": char_names[0], "scenario": "balanced",
            "items": fallback, "rationale": "通用角色出装推荐。",
        }]}

    # 构建装备库描述（给 LLM 的上下文）
    items_info = "\n".join(
        f"- {name} ({info.get('category', '')}): {info.get('stat', '')}, {info.get('note', '')}"
        for name, info in all_items.items()
    )

    # LLM 出装推荐
    prompt = f"""You are a build advisor. Recommend an equipment set for {', '.join(char_names)} (role: {role}).

Available items:
{items_info}

Reply in this format:
Build: Item1 -> Item2 -> Item3 -> Item4 -> Item5 -> Item6
Rationale: (2-3 sentences why)
Scenario: (balanced / ahead / behind)
Respond in the same language as the player's question."""

    try:
        result = llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        import re
        items_match = re.search(r"Build:\s*(.+)", text)
        items_str = items_match.group(1) if items_match else ""
        items_list = [i.strip() for i in items_str.split("->") if i.strip()]
        rationale_match = re.search(r"Rationale:\s*(.+)", text)
        rationale = rationale_match.group(1) if rationale_match else text[:100]
        scenario_match = re.search(r"Scenario:\s*(.+)", text)
        scenario = scenario_match.group(1).strip() if scenario_match else "balanced"
        return {"build_recommendations": [{
            "character": char_names[0], "scenario": scenario,
            "items": items_list, "rationale": rationale,
        }]}
    except Exception:
        logger.exception("LLM build recommendation failed")
        return {"build_recommendations": []}


# ═══════════════════════════════════════════════════════════════
# RAG 检索
# ═══════════════════════════════════════════════════════════════

def rag_agent(state: GameCoachState) -> GameCoachState:
    """
    攻略检索代理。

    将用户问题 + 战绩分析中识别到的弱点合并为检索查询，
    通过 FAISS 向量检索从 data/guides/ 找到相关攻略片段。

    检索结果（rag_context）会注入到 strategy_agent 的 prompt 中，
    使策略建议有攻略依据而非凭空生成。

    RAG 模块不可用时（DashScope Key 未配 / FAISS 索引文件缺失），
    静默返回空列表，不阻断后续节点。
    """
    user_msg = state.get("normalized_message", state.get("user_message", ""))
    analysis = state.get("match_analysis", {})

    # 构建增强检索查询：用户问题 + 已识别的弱点 → 更精准的检索
    query_parts = [user_msg]
    for w in analysis.get("weaknesses", []):
        query_parts.append(w)
    query = " ".join(query_parts)

    try:
        from gamecoach.rag.retriever import retrieve
        docs = retrieve(query, top_k=5)
    except ImportError:
        logger.warning("RAG module not available")
        docs = []
    except Exception:
        logger.exception("RAG retrieval failed")
        docs = []

    return {"rag_context": docs}


# ═══════════════════════════════════════════════════════════════
# 策略生成
# ═══════════════════════════════════════════════════════════════

def strategy_agent(state: GameCoachState) -> GameCoachState:
    """
    策略生成代理。

    委托给 create_llm_strategy（LLM / fallback），
    综合战绩分析、玩家画像、角色推荐和 RAG 结果，
    生成个性化策略建议和训练计划。

    执行后自动回写 memory：将新发现的弱点追加到玩家画像。
    """
    result = create_llm_strategy(state)

    # 将分析中发现的弱点写回 memory（非致命操作）
    strategy = result.get("strategy", {})
    if strategy.get("weaknesses"):
        try:
            update_player_memory(state.get("player_id", "player_001"), {
                "weaknesses": strategy["weaknesses"],
                "source": "match_analysis",
            })
        except Exception:
            logger.debug("Memory update skipped (non-fatal)")
    return result


# ═══════════════════════════════════════════════════════════════
# 回复合成
# ═══════════════════════════════════════════════════════════════

def response_agent(state: GameCoachState) -> GameCoachState:
    """
    回复合成代理。

    委托给 create_llm_response（LLM / fallback），
    汇总所有 Agent 的结果，生成面向玩家的结构化建议。

    输出格式：结论 → 数据依据 → 优先改进 → 角色推荐 → 训练计划。
    """
    return create_llm_response(state)


# ═══════════════════════════════════════════════════════════════
# 评估日志
# ═══════════════════════════════════════════════════════════════

def evaluation_logger(state: GameCoachState) -> GameCoachState:
    """
    评估日志记录。

    DAG 的最后一个节点。从 state 中提取关键指标，
    计算路由路径、降级节点数、RAG 命中数等评估数据。

    这些指标用于：
    - LangSmith 自定义 feedback 上报（evaluation/metrics.py）
    - CLI 输出（main.py 的 _run_and_print）
    - 后续离线评估和趋势监控
    """
    routing = state.get("routing_decisions", {})
    # 提取实际执行了的节点
    executed = [k for k, v in routing.items() if v in ("execute", "fixed")]

    metrics = {
        "planner_task_count": len(state.get("planned_tasks", [])),
        "executed_nodes": executed,
        "routing_path": " -> ".join(executed),  # 可视化执行路径
        "degraded_node_count": len(state.get("degraded_nodes", [])),
        "tool_call_success_rate": 1.0,  # Mock 工具始终成功
        "has_match_analysis": bool(state.get("match_analysis", {}).get("metrics")),
        "has_character_recommendations": bool(state.get("character_recommendations")),
        "has_build_recommendations": bool(state.get("build_recommendations")),
        "has_training_plan": bool(state.get("training_plan", {}).get("daily_tasks")),
        "rag_hit_count": len(state.get("rag_context", [])),
        "response_length": len(state.get("final_response", "")),
    }
    return {"metrics": metrics}
