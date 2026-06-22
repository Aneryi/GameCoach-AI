"""
任务路由器 — 将 Planner 的任务列表转化为 LangGraph 执行路径。

工作原理：
1. Planner 生成 planned_tasks（如 [match_analysis, character_recommendation]）
2. TASK_NODE_MAP 把 task_type 映射为实际的节点名
3. EXECUTION_ORDER 定义节点在 DAG 中的固定顺序
4. 条件边函数（route_after_*）根据 planned_tasks 决定下一个节点

为什么 routing_decisions 在 planner 节点中计算而非条件边函数中：
LangGraph 的条件边函数（如 route_after_planner）接收的 state 是快照副本，
函数内的修改不会被写回。只有节点函数（返回 dict）才能更新 state。
因此 routing_decisions 在 planner 节点返回中设置。

核心数据结构：
- TASK_NODE_MAP:  Planner 的 task_type → 实际执行的节点名
- EXECUTION_ORDER: 数据节点的 DAG 执行顺序（顺序有依赖关系）
- FIXED_NODES:     始终执行的节点（策略合成、回复、评估）
"""

from gamecoach.graph.state import GameCoachState

# ── 任务类型到节点名的映射 ──
# key: planned_tasks 中的 task_type 值
# value: workflow.py 中 add_node 注册的节点名
TASK_NODE_MAP: dict[str, str] = {
    "match_analysis": "match_analysis_agent",
    "character_recommendation": "character_recommendation_agent",
    "build_recommendation": "build_agent",
    "strategy_generation": "strategy_agent",
    "rag_lookup": "rag_agent",
    "memory_lookup": "memory_loader",
    "training_plan": "strategy_agent",  # 训练计划由 strategy_agent 一并生成
}

# ── 数据节点的 DAG 执行顺序 ──
# 顺序有依赖关系：
#   1. memory_loader: 必须最先（所有 Agent 都需要玩家画像）
#   2. match_analysis: 战绩数据不依赖其他数据节点
#   3. character_recommendation: 需要 memory（玩家偏好）+ patch meta（版本信息）
#   4. rag_agent: 需要 match_analysis（用弱点改写查询词）+ character pool（检索方向）
#   5. build_agent: 需要 character_recommendation（为推荐的角色配装）
EXECUTION_ORDER = [
    "memory_loader",
    "match_analysis_agent",
    "character_recommendation_agent",
    "rag_agent",
    "build_agent",
]

# ── 始终执行的节点 ──
# 这些节点在 DAG 末尾，不管 planned_tasks 中是否列出，都会执行
FIXED_NODES = ["strategy_agent", "response_agent", "evaluation_logger"]


def _collect_required_nodes(state: GameCoachState) -> list[str]:
    """
    从 planned_tasks 中提取需要执行的节点名列表。

    遍历 planned_tasks，通过 TASK_NODE_MAP 映射为节点名，
    去重后返回。顺序由 planned_tasks 决定（已按 priority 排序）。

    Args:
        state: 当前状态（包含 planned_tasks）。

    Returns:
        需要执行的节点名列表，如 ["memory_loader", "match_analysis_agent", ...]。
        如果 planned_tasks 为空则返回 []。
    """
    planned = state.get("planned_tasks", [])
    if not planned:
        return []

    seen: set[str] = set()
    ordered: list[str] = []
    for task in planned:
        node = TASK_NODE_MAP.get(task["task_type"])
        if node and node not in seen:
            seen.add(node)
            ordered.append(node)
    return ordered


def route_after_planner(state: GameCoachState) -> str:
    """
    Planner 之后的第一个节点路由。

    逻辑：
    - 无任务 → 直接到 strategy_agent（跳过所有数据节点）
    - 有任务 → 从 EXECUTION_ORDER 中找到第一个需要的节点
    - memory_loader 在任何有任务的情况下都优先执行（其他 Agent 依赖玩家画像）

    Args:
        state: 当前状态（包含 planned_tasks）。

    Returns:
        下一个要执行的节点名。
    """
    required = _collect_required_nodes(state)
    if not required:
        return "strategy_agent"
    if "memory_loader" in required:
        return "memory_loader"
    return required[0]


def route_after_memory(state: GameCoachState) -> str:
    """memory_loader 之后的下一个数据节点。"""
    return _next_in_order("memory_loader", _collect_required_nodes(state))


def route_after_match_analysis(state: GameCoachState) -> str:
    """match_analysis_agent 之后的下一个数据节点。"""
    return _next_in_order("match_analysis_agent", _collect_required_nodes(state))


def route_after_character_recommendation(state: GameCoachState) -> str:
    """character_recommendation_agent 之后的下一个数据节点。"""
    return _next_in_order("character_recommendation_agent", _collect_required_nodes(state))


def route_after_rag(state: GameCoachState) -> str:
    """rag_agent 之后的下一个数据节点。"""
    return _next_in_order("rag_agent", _collect_required_nodes(state))


def route_after_build(state: GameCoachState) -> str:
    """build_agent 之后全部数据节点执行完毕，进入 strategy_agent。"""
    return _next_in_order("build_agent", _collect_required_nodes(state))


def _next_in_order(current: str, required: list[str]) -> str:
    """
    在 EXECUTION_ORDER 中找到 current 之后下一个需要执行的节点。

    遍历 EXECUTION_ORDER，跳过 current 之前的，找到 current 之后
    第一个在 required 列表中出现的节点。如果后面没有了，返回 strategy_agent。

    Args:
        current: 当前刚执行完的节点名。
        required: 需要执行的节点名列表。

    Returns:
        下一个节点名。如果所有数据节点都已执行，返回 "strategy_agent"。
    """
    try:
        idx = EXECUTION_ORDER.index(current)
    except ValueError:
        # current 不在 EXECUTION_ORDER 中（如 FIXED_NODES）→ 直接到 strategy
        return "strategy_agent"

    for candidate in EXECUTION_ORDER[idx + 1:]:
        if candidate in required:
            return candidate

    # 所有条件数据节点执行完毕 → 进入策略合成阶段
    return "strategy_agent"
