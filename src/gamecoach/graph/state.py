"""
LangGraph 全局状态定义。

GameCoachState 是贯穿整个工作流的状态容器。每个 LangGraph 节点
读取 state 中的输入字段，返回要更新的字段（partial update）。
TypedDict 的 total=False 使所有字段可选，每个节点只需返回自己产生的字段。

字段分类：
- 输入层: user_message, normalized_message, player_id
- Planner 层: intent, planned_tasks
- 路由层: routing_decisions
- 数据层: memory, match_data, match_analysis, character_recommendations, build_recommendations, rag_context
- 策略层: strategy, training_plan
- 输出层: final_response
- 监控层: errors, degraded_nodes, metrics
"""

from typing import Any, Literal, TypedDict


class PlayerMemory(TypedDict, total=False):
    """
    玩家长期画像。

    存储玩家的游戏偏好、能力短板和成长目标。
    数据从 data/test_data/player_memory.json 加载，
    在 strategy_agent 执行后自动回写更新。

    字段通过 total=False 全部设为可选——如果某个玩家还没设置某个偏好，该字段就不存在。
    """
    player_id: str
    favorite_characters: list[str]    # 偏好角色，如 ["Delta", "Alpha"]
    main_roles: list[str]             # 主玩位置，如 ["damage"]
    weaknesses: list[str]             # 能力短板（来自数据分析和玩家自述）
    goals: list[str]                  # 上分目标，如 ["reach Diamond rank"]
    preferred_playstyle: str          # 打法风格，如 "conservative farming"
    rank: str                         # 当前段位，如 "Platinum I"
    updated_at: str                   # ISO 时间戳，记录最后更新时间


class PlannedTask(TypedDict):
    """
    Planner 拆解出的单个子任务。

    task_type 限定为 7 种预定义类型（Literal 枚举），防止 LLM 发明不存在的任务。
    required_tools 列出该任务依赖的工具名——这些工具必须在 tools/ 中注册为 @tool。
    """
    task_id: str                      # 任务编号，如 "t1", "t2"
    task_type: Literal[               # 任务类型（枚举，LLM 不能发明新类型）
        "match_analysis",             # 战绩分析
        "character_recommendation",   # 角色推荐
        "build_recommendation",       # 出装推荐
        "strategy_generation",        # 策略生成
        "rag_lookup",                 # 攻略检索
        "memory_lookup",              # 读取玩家画像
        "training_plan",              # 训练计划生成
    ]
    description: str                  # 一句话描述任务要做什么
    priority: int                     # 优先级，1 最高
    required_tools: list[str]         # 依赖的工具名列表


class GameCoachState(TypedDict, total=False):
    """
    LangGraph 工作流的全局状态。

    total=False 意味着所有字段都是可选的。每个节点函数返回一个 dict，
    LangGraph 自动将返回的 dict 合并到 state 中（partial update），
    不需要在节点中返回完整的 state。

    key 是字段名，value 的类型可以在节点函数中通过 state.get("key") 获取。
    """
    # ── 输入层 ──
    user_message: str                 # 原始用户输入
    normalized_message: str           # 标准化后的输入（去首尾空格）
    player_id: str                    # 玩家 ID，默认 "player_001"

    # ── Planner 层 ──
    intent: str                       # 用户意图分类，如 "improve_performance"
    planned_tasks: list[PlannedTask]  # Planner 拆解的任务列表

    # ── 路由层 ──
    routing_decisions: dict[str, str] # 每个节点的路由决策: "execute" / "skip" / "fixed"

    # ── 数据层 ──
    memory: PlayerMemory              # 玩家长期画像
    match_data: dict[str, Any]        # 战绩原始数据（从工具获取）
    match_analysis: dict[str, Any]    # 战绩分析结果（统计计算后）
    character_recommendations: list[dict[str, Any]]  # 角色推荐列表
    build_recommendations: list[dict[str, Any]]      # 出装推荐列表
    rag_context: list[dict[str, Any]] # RAG 检索到的攻略片段

    # ── 策略层 ──
    strategy: dict[str, Any]          # 策略建议（诊断、优先级、动作项）
    training_plan: dict[str, Any]     # 训练计划（天数、每日任务、成功标准）

    # ── 输出层 ──
    final_response: str               # 最终返回给玩家的完整建议

    # ── 监控层 ──
    errors: list[str]                 # 执行过程中的非致命错误
    degraded_nodes: list[str]         # 被降级/跳过的节点名列表
    metrics: dict[str, Any]           # 评估指标（planner_task_count 等）
