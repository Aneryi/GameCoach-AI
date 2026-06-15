"""评估指标计算。

从 GameCoachState 中提取关键质量指标，评估本次 Agent 执行的质量。
"""

from __future__ import annotations

from gamecoach.graph.state import GameCoachState


def evaluate_run(state: GameCoachState) -> dict:
    """计算本次执行的质量评估报告。

    Args:
        state: 完整的 GameCoachState。

    Returns:
        评估报告字典，包含各维度的评分和诊断。
    """
    metrics = state.get("metrics", {})
    routing = state.get("routing_decisions", {})
    errors = state.get("errors", [])
    degraded = state.get("degraded_nodes", [])

    # 基本指标
    planned_count = len(state.get("planned_tasks", []))
    executed_count = len([v for v in routing.values() if v in ("execute", "fixed")])

    report = {
        "execution": {
            "planned_tasks": planned_count,
            "executed_nodes": executed_count,
            "skipped_nodes": len([v for v in routing.values() if v == "skip"]),
            "degraded_nodes": len(degraded),
            "errors": len(errors),
        },
        "data_quality": {
            "has_match_data": metrics.get("has_match_analysis", False),
            "has_hero_recs": metrics.get("has_hero_recommendations", False),
            "has_build_recs": metrics.get("has_build_recommendations", False),
            "has_training_plan": metrics.get("has_training_plan", False),
            "rag_hits": metrics.get("rag_hit_count", 0),
        },
        "output_quality": {
            "response_length": metrics.get("response_length", 0),
            "has_conclusion": "结论" in state.get("final_response", ""),
            "has_data_evidence": "胜率" in state.get("final_response", ""),
            "has_action_items": "优先改进" in state.get("final_response", "")
            or "改进" in state.get("final_response", ""),
        },
        "health": _compute_health(planned_count, executed_count, len(errors), len(degraded)),
    }

    return report


def _compute_health(
    planned: int,
    executed: int,
    error_count: int,
    degraded_count: int,
) -> str:
    """综合评估执行健康度。"""
    if error_count > 0:
        return "WARN: 执行中有错误"
    if degraded_count > 2:
        return "WARN: 多个节点降级，建议检查服务可用性"
    if planned == 0:
        return "WARN: Planner 未生成任务"
    if executed == 0:
        return "FAIL: 无节点执行"
    if degraded_count > 0:
        return "OK: 部分降级但可用"
    return "OK: 健康"
