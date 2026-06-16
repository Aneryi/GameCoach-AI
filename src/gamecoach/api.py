"""GameCoach AI FastAPI server."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from gamecoach.graph.workflow import graph

logger = logging.getLogger(__name__)

app = FastAPI(title="GameCoach AI", description="LangGraph-based game coaching agent API", version="0.2.0")


class CoachRequest(BaseModel):
    user_message: str = Field(description="Player input", examples=["My win rate is low, how can I climb?"])
    player_id: str = Field(default="player_001")


class MetricsSummary(BaseModel):
    planner_task_count: int = 0
    routing_path: str = ""
    degraded_node_count: int = 0
    rag_hit_count: int = 0
    response_length: int = 0
    has_match_analysis: bool = False
    has_character_recommendations: bool = False
    has_build_recommendations: bool = False
    has_training_plan: bool = False


class CoachResponse(BaseModel):
    status: str = Field(description="ok / error")
    intent: Optional[str] = None
    planned_tasks: list[dict] = []
    routing_path: str = ""
    final_response: str = ""
    metrics: MetricsSummary = MetricsSummary()
    errors: list[str] = []
    degraded_nodes: list[str] = []


@app.get("/")
def root():
    return {"service": "GameCoach AI", "version": "0.2.0", "status": "healthy"}


@app.post("/coach", response_model=CoachResponse)
def coach(request: CoachRequest):
    graph_input = {"user_message": request.user_message, "player_id": request.player_id}
    try:
        result = graph.invoke(graph_input)
    except Exception:
        logger.exception("Graph execution failed")
        return CoachResponse(status="error", errors=["System error, please retry."])

    m = result.get("metrics", {})
    return CoachResponse(
        status="ok",
        intent=result.get("intent"),
        planned_tasks=result.get("planned_tasks", []),
        routing_path=m.get("routing_path", ""),
        final_response=result.get("final_response", ""),
        metrics=MetricsSummary(
            planner_task_count=m.get("planner_task_count", 0),
            routing_path=m.get("routing_path", ""),
            degraded_node_count=m.get("degraded_node_count", 0),
            rag_hit_count=m.get("rag_hit_count", 0),
            response_length=m.get("response_length", 0),
            has_match_analysis=m.get("has_match_analysis", False),
            has_character_recommendations=m.get("has_character_recommendations", False),
            has_build_recommendations=m.get("has_build_recommendations", False),
            has_training_plan=m.get("has_training_plan", False),
        ),
        errors=result.get("errors", []),
        degraded_nodes=result.get("degraded_nodes", []),
    )


@app.get("/demo")
def demo_scenarios():
    return {"scenarios": [
        {"id": 0, "name": "Rank climbing", "message": "My win rate is low and I keep dying in teamfights. How can I climb?"},
        {"id": 1, "name": "Character recommendation", "message": "What characters are strong in the current meta?"},
        {"id": 2, "name": "Build advice", "message": "What's the best build for Alpha?"},
        {"id": 3, "name": "Training plan", "message": "Create a 7-day training plan for me."},
    ]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
