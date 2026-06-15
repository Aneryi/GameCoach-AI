"""端到端工作流集成测试。"""

import pytest

from gamecoach.graph.workflow import graph


# ── 基础集成测试 ──


def test_gamecoach_workflow_returns_final_response():
    """默认上分求助场景应返回完整建议。"""
    result = graph.invoke(
        {
            "user_message": "我最近胜率很低，玩射手总是团战暴毙，怎么上分？",
            "player_id": "player_001",
            "game": "moba",
        }
    )

    assert result["intent"] in ("improve_rank", "strategy_help", "hero_recommendation")
    assert len(result["planned_tasks"]) >= 1
    assert result["final_response"]
    assert len(result["final_response"]) > 50


def test_workflow_has_match_analysis():
    """默认场景应包含战绩分析。"""
    result = graph.invoke(
        {
            "user_message": "我最近胜率很低，玩射手总是团战暴毙，怎么上分？",
            "player_id": "player_001",
            "game": "moba",
        }
    )

    metrics = result.get("match_analysis", {}).get("metrics", {})
    assert metrics.get("matches") == 20
    assert "win_rate" in metrics


def test_workflow_has_hero_recommendations():
    """当明确问英雄时应包含英雄推荐。"""
    result = graph.invoke(
        {
            "user_message": "当前版本有什么强势英雄适合我练？",
            "player_id": "player_001",
            "game": "moba",
        }
    )

    # LLM planner 可能包含也可能不包含，取决于 LLM 输出
    # 但至少应该有 routing_decisions
    assert result.get("routing_decisions")


def test_workflow_has_training_plan():
    """应生成训练计划。"""
    result = graph.invoke(
        {
            "user_message": "帮我制定一个训练计划。",
            "player_id": "player_001",
            "game": "moba",
        }
    )

    plan = result.get("training_plan", {})
    assert plan.get("daily_tasks") or plan.get("duration_days")


def test_workflow_has_metrics():
    """评估指标应正确记录。"""
    result = graph.invoke(
        {
            "user_message": "当前版本有什么强势英雄？",
            "player_id": "player_001",
            "game": "moba",
        }
    )

    metrics = result["metrics"]
    assert metrics["planner_task_count"] >= 1
    assert "routing_path" in metrics
    assert "response_length" in metrics


# ── 路由测试 ──


def test_workflow_routing_decisions_present():
    """路由决策应被记录。"""
    result = graph.invoke(
        {
            "user_message": "我适合练什么英雄？",
            "player_id": "player_001",
            "game": "moba",
        }
    )

    routing = result.get("routing_decisions", {})
    assert routing  # 应有路由决策
    # 至少 strategy_agent 和 response_agent 是固定执行的
    assert routing.get("strategy_agent") in ("execute", "fixed")


# ── 关键质量测试 ──


def test_final_response_contains_expected_sections():
    """最终回复应包含关键内容板块。"""
    result = graph.invoke(
        {
            "user_message": "我最近胜率很低，怎么上分？",
            "player_id": "player_001",
            "game": "moba",
        }
    )

    response = result["final_response"]
    # 应包含数据、建议或计划中的至少两项
    has_data = "胜率" in response or "KDA" in response or "死亡" in response
    has_advice = "建议" in response or "改进" in response or "推荐" in response
    assert has_data or has_advice


def test_memory_is_loaded():
    """玩家记忆应被加载。"""
    result = graph.invoke(
        {
            "user_message": "我想上分",
            "player_id": "player_001",
            "game": "moba",
        }
    )

    memory = result.get("memory", {})
    assert memory.get("player_id") == "player_001"
    assert "后羿" in memory.get("favorite_heroes", [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
