"""任务路由器 —— 根据 Planner 输出的任务列表决定执行哪些节点。"""

from gamecoach.graph.state import GameCoachState

TASK_NODE_MAP: dict[str, str] = {
    "match_analysis": "match_analysis_agent",
    "character_recommendation": "character_recommendation_agent",
    "build_recommendation": "build_agent",
    "strategy_generation": "strategy_agent",
    "rag_lookup": "rag_agent",
    "memory_lookup": "memory_loader",
    "training_plan": "strategy_agent",
}

EXECUTION_ORDER = [
    "memory_loader",
    "match_analysis_agent",
    "character_recommendation_agent",
    "rag_agent",
    "build_agent",
]

FIXED_NODES = ["strategy_agent", "response_agent", "evaluation_logger"]


def _collect_required_nodes(state: GameCoachState) -> list[str]:
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
    required = _collect_required_nodes(state)
    if not required:
        return "strategy_agent"
    if "memory_loader" in required:
        return "memory_loader"
    return required[0]


def route_after_memory(state: GameCoachState) -> str:
    return _next_in_order("memory_loader", _collect_required_nodes(state))


def route_after_match_analysis(state: GameCoachState) -> str:
    return _next_in_order("match_analysis_agent", _collect_required_nodes(state))


def route_after_character_recommendation(state: GameCoachState) -> str:
    return _next_in_order("character_recommendation_agent", _collect_required_nodes(state))


def route_after_rag(state: GameCoachState) -> str:
    return _next_in_order("rag_agent", _collect_required_nodes(state))


def route_after_build(state: GameCoachState) -> str:
    return _next_in_order("build_agent", _collect_required_nodes(state))


def _next_in_order(current: str, required: list[str]) -> str:
    try:
        idx = EXECUTION_ORDER.index(current)
    except ValueError:
        return "strategy_agent"
    for candidate in EXECUTION_ORDER[idx + 1:]:
        if candidate in required:
            return candidate
    return "strategy_agent"
