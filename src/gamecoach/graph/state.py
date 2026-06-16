from typing import Any, Literal, TypedDict


class PlayerMemory(TypedDict, total=False):
    player_id: str
    favorite_characters: list[str]
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
        "character_recommendation",
        "build_recommendation",
        "strategy_generation",
        "rag_lookup",
        "memory_lookup",
        "training_plan",
    ]
    description: str
    priority: int
    required_tools: list[str]


class GameCoachState(TypedDict, total=False):
    # 输入
    user_message: str
    normalized_message: str
    player_id: str
    intent: str

    # Planner 输出
    planned_tasks: list[PlannedTask]

    # 路由决策
    routing_decisions: dict[str, str]

    # Memory
    memory: PlayerMemory

    # 战绩分析
    match_data: dict[str, Any]
    match_analysis: dict[str, Any]

    # 角色推荐
    character_recommendations: list[dict[str, Any]]

    # 出装推荐
    build_recommendations: list[dict[str, Any]]

    # RAG 检索
    rag_context: list[dict[str, Any]]

    # 策略与训练计划
    strategy: dict[str, Any]
    training_plan: dict[str, Any]

    # 最终输出
    final_response: str

    # 错误与降级
    errors: list[str]
    degraded_nodes: list[str]

    # 评估指标
    metrics: dict[str, Any]
