"""End-to-end workflow integration tests."""

import pytest

from gamecoach.graph.workflow import graph


def test_gamecoach_workflow_returns_final_response():
    result = graph.invoke({"user_message": "My win rate is low and I keep dying in teamfights. How can I climb?", "player_id": "player_001"})
    assert result["final_response"]
    assert len(result["final_response"]) > 50


def test_workflow_has_match_analysis():
    result = graph.invoke({"user_message": "My win rate is low, how can I climb?", "player_id": "player_001"})
    metrics = result.get("match_analysis", {}).get("metrics", {})
    assert metrics.get("matches") == 20
    assert "win_rate" in metrics


def test_workflow_has_character_recommendations():
    result = graph.invoke({"user_message": "What characters are strong in the current meta?", "player_id": "player_001"})
    assert result.get("routing_decisions")


def test_workflow_has_training_plan():
    result = graph.invoke({"user_message": "Create a training plan for me.", "player_id": "player_001"})
    plan = result.get("training_plan", {})
    assert plan.get("daily_tasks") or plan.get("duration_days")


def test_workflow_has_metrics():
    result = graph.invoke({"user_message": "What characters are strong right now?", "player_id": "player_001"})
    metrics = result["metrics"]
    assert metrics["planner_task_count"] >= 1
    assert "routing_path" in metrics


def test_workflow_routing_decisions_present():
    result = graph.invoke({"user_message": "What character should I learn?", "player_id": "player_001"})
    routing = result.get("routing_decisions", {})
    assert routing


def test_memory_is_loaded():
    result = graph.invoke({"user_message": "I want to climb", "player_id": "player_001"})
    memory = result.get("memory", {})
    assert memory.get("player_id") == "player_001"
    assert "Delta" in memory.get("favorite_characters", [])
