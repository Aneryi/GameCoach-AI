from typing import Any, Literal, TypedDict


class PlayerMemory(TypedDict, total=False):
    player_id: str
    favorite_heroes: list[str]
    main_roles: list[str]
    weaknesses: list[str]
    goals: list[str]
    preferred_playstyle: str
    rank: str
    updated_at: str


class PlannedTask(TypedDict):
    task_id: str
    task_type: Literal[
        "match_analysis",
        "hero_recommendation",
        "strategy_generation",
        "memory_lookup",
        "training_plan",
    ]
    description: str
    priority: int
    required_tools: list[str]


class GameCoachState(TypedDict, total=False):
    user_message: str
    normalized_message: str
    player_id: str
    game: str
    intent: str
    memory: PlayerMemory
    planned_tasks: list[PlannedTask]
    match_data: dict[str, Any]
    match_analysis: dict[str, Any]
    hero_recommendations: list[dict[str, Any]]
    strategy: dict[str, Any]
    training_plan: dict[str, Any]
    final_response: str
    errors: list[str]
    metrics: dict[str, Any]

