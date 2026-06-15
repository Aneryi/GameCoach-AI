"""任务路由器 —— 根据 Planner 输出的任务列表决定执行哪些节点。

LangGraph 的条件边通过本模块的路由函数来决定流程走向，
跳过不需要的节点（例如用户只问英雄推荐时不必跑战绩分析）。
"""

from gamecoach.graph.state import GameCoachState

# task_type → 负责处理的节点名
TASK_NODE_MAP: dict[str, str] = {
    "match_analysis": "match_analysis_agent",
    "hero_recommendation": "hero_recommendation_agent",
    "build_recommendation": "build_agent",
    "strategy_generation": "strategy_agent",
    "rag_lookup": "rag_agent",
    "memory_lookup": "memory_loader",
    "training_plan": "strategy_agent",
}

# DAG 执行顺序：数据节点按此顺序依次检查是否需要执行
EXECUTION_ORDER = [
    "memory_loader",
    "match_analysis_agent",
    "hero_recommendation_agent",
    "build_agent",
    "rag_agent",
]

# 始终执行的节点（在所有数据节点之后）
FIXED_NODES = ["strategy_agent", "response_agent", "evaluation_logger"]


def _collect_required_nodes(state: GameCoachState) -> list[str]:
    """从 planned_tasks 中提取需要执行的节点列表，保持 DAG 顺序。"""
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
    """Planner 之后的第一个数据节点。

    始终先加载 memory（如果有任何任务），然后按 EXECUTION_ORDER 依次路由。
    """
    required = _collect_required_nodes(state)

    if not required:
        return "strategy_agent"

    # 始终先走 memory_loader（如果有任何数据任务需要）
    if "memory_loader" in required:
        return "memory_loader"

    # 否则走第一个需要的数据节点
    return required[0]


def route_after_memory(state: GameCoachState) -> str:
    """Memory 加载之后，决定下一个数据节点。"""
    required = _collect_required_nodes(state)
    return _next_in_order("memory_loader", required)


def route_after_match_analysis(state: GameCoachState) -> str:
    """战绩分析之后。"""
    required = _collect_required_nodes(state)
    return _next_in_order("match_analysis_agent", required)


def route_after_hero_recommendation(state: GameCoachState) -> str:
    """英雄推荐之后。"""
    required = _collect_required_nodes(state)
    return _next_in_order("hero_recommendation_agent", required)


def route_after_rag(state: GameCoachState) -> str:
    """RAG 之后。"""
    required = _collect_required_nodes(state)
    return _next_in_order("rag_agent", required)


def route_after_build(state: GameCoachState) -> str:
    """出装推荐之后。"""
    required = _collect_required_nodes(state)
    return _next_in_order("build_agent", required)


def _next_in_order(current: str, required: list[str]) -> str:
    """找到 EXECUTION_ORDER 中 current 之后下一个需要的节点。

    如果后面没有需要的数据节点了，回退到 strategy_agent。
    """
    try:
        idx = EXECUTION_ORDER.index(current)
    except ValueError:
        return "strategy_agent"

    for candidate in EXECUTION_ORDER[idx + 1:]:
        if candidate in required:
            return candidate

    # 所有条件数据节点跑完，到策略合成
    return "strategy_agent"
