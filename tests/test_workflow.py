from gamecoach.graph.workflow import graph


def test_gamecoach_workflow_returns_final_response():
    result = graph.invoke(
        {
            "user_message": "我最近胜率很低，玩射手总是团战暴毙，怎么上分？",
            "player_id": "player_001",
            "game": "moba",
        }
    )

    assert result["intent"] == "improve_rank"
    assert result["match_analysis"]["metrics"]["matches"] == 20
    assert result["hero_recommendations"]
    assert "推荐英雄" in result["final_response"]
    assert result["metrics"]["has_training_plan"] is True

