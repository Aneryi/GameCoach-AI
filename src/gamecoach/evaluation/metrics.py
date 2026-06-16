"""评估指标计算与 LangSmith 上报。

从 GameCoachState 中提取关键质量指标，评估本次 Agent 执行的质量，
并通过 LangSmith SDK 上报自定义指标（如果 API Key 已配置）。
"""

from __future__ import annotations

import logging
import os

from gamecoach.graph.state import GameCoachState

logger = logging.getLogger(__name__)

# LangSmith client 惰性初始化
_langsmith_client = None


def _get_langsmith_client():
    """获取 LangSmith client 实例。

    如果 LANGCHAIN_API_KEY 未配置则返回 None，所有上报操作静默跳过。
    """
    global _langsmith_client
    if _langsmith_client is None:
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if not api_key:
            return None
        try:
            from langsmith import Client
            _langsmith_client = Client(api_key=api_key)
        except ImportError:
            logger.debug("langsmith SDK 未安装，跳过 LangSmith 上报")
            return None
        except Exception:
            logger.debug("LangSmith client 初始化失败")
            return None
    return _langsmith_client


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
        },
        "health": _compute_health(planned_count, executed_count, len(errors), len(degraded)),
    }

    # 尝试上报到 LangSmith
    _push_to_langsmith(report, state)

    return report


def _push_to_langsmith(report: dict, state: GameCoachState) -> None:
    """将评估指标上报到 LangSmith。

    通过 LangSmith SDK 创建自定义 feedback，
    在 LangSmith UI 中可查看每次请求的质量趋势。
    """
    client = _get_langsmith_client()
    if client is None:
        return

    project = os.getenv("LANGCHAIN_PROJECT", "gamecoach-ai")

    try:
        client.create_feedback(
            key="planner_task_count",
            score=report["execution"]["planned_tasks"],
            comment=f"任务数: {report['execution']['planned_tasks']}",
            project_name=project,
        )
        client.create_feedback(
            key="degraded_nodes",
            score=float(report["execution"]["degraded_nodes"]),
            comment=f"降级节点: {report['execution']['degraded_nodes']}",
            project_name=project,
        )
        client.create_feedback(
            key="rag_hits",
            score=float(report["data_quality"]["rag_hits"]),
            comment=f"RAG 命中: {report['data_quality']['rag_hits']}",
            project_name=project,
        )
        client.create_feedback(
            key="response_length",
            score=float(report["output_quality"]["response_length"]),
            comment=f"回复长度: {report['output_quality']['response_length']}",
            project_name=project,
        )
        health_score = 1.0 if "OK" in report["health"] else 0.5
        client.create_feedback(
            key="health",
            score=health_score,
            comment=report["health"],
            project_name=project,
        )
        logger.debug("LangSmith 上报完成 (project=%s)", project)
    except Exception:
        logger.debug("LangSmith 上报失败（非致命）")


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
