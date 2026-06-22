"""
LangGraph 多 Agent 工作流编排。

构建带条件路由的状态图（StateGraph），定义各 Agent 节点的执行顺序
和条件分支。编译后的 graph 可通过 graph.invoke(input_state) 执行。

DAG 结构：
    START → input_normalizer → planner
                                  |
                        [条件路由: route_after_planner]
                                  |
                            memory_loader
                                  |
                          match_analysis
                                  |
                     character_recommendation
                                  |
                             rag_agent
                                  |
                            build_agent
                                  |
                          strategy_agent  ← 固定执行
                                  |
                          response_agent ← 固定执行
                                  |
                        evaluation_logger → END

条件边 vs 固定边：
- 固定边（add_edge）: 前驱执行完 → 后继必定执行
- 条件边（add_conditional_edges）: 前驱执行完 → 调用路由函数 → 根据返回值选择后继
  条件边的第三个参数是路由函数可能返回值的映射表（LangGraph 要求显式声明）。
"""

from langgraph.graph import END, START, StateGraph

from gamecoach.graph.nodes import (
    build_agent,
    character_recommendation_agent,
    evaluation_logger,
    input_normalizer,
    match_analysis_agent,
    memory_loader,
    planner,
    rag_agent,
    response_agent,
    strategy_agent,
)
from gamecoach.graph.router import (
    route_after_build,
    route_after_character_recommendation,
    route_after_match_analysis,
    route_after_memory,
    route_after_planner,
    route_after_rag,
)
from gamecoach.graph.state import GameCoachState


def build_graph():
    """
    构建并编译 LangGraph 状态图。

    Returns:
        编译后的 CompiledStateGraph，可直接调用 .invoke() 执行。
    """
    # StateGraph: LangGraph 的核心抽象，节点共享同一份 GameCoachState
    workflow = StateGraph(GameCoachState)

    # ═══════════════════════════════════════════════════════════
    # 注册节点
    # add_node(name, function): 将函数注册为图中的一个节点
    # 节点名用于后续的 add_edge / add_conditional_edges
    # ═══════════════════════════════════════════════════════════
    workflow.add_node("input_normalizer", input_normalizer)
    workflow.add_node("planner", planner)
    workflow.add_node("memory_loader", memory_loader)
    workflow.add_node("match_analysis_agent", match_analysis_agent)
    workflow.add_node("character_recommendation_agent", character_recommendation_agent)
    workflow.add_node("build_agent", build_agent)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("strategy_agent", strategy_agent)
    workflow.add_node("response_agent", response_agent)
    workflow.add_node("evaluation_logger", evaluation_logger)

    # ═══════════════════════════════════════════════════════════
    # 固定边：输入 → planner
    # START 是 LangGraph 内置的虚拟起始节点
    # ═══════════════════════════════════════════════════════════
    workflow.add_edge(START, "input_normalizer")
    workflow.add_edge("input_normalizer", "planner")

    # ═══════════════════════════════════════════════════════════
    # 条件边：数据节点层（按需执行）
    # add_conditional_edges(source, route_fn, route_map)
    #   - source: 条件边的起点
    #   - route_fn: 路由函数，接收 state，返回一个路由键（str）
    #   - route_map: 路由键 → 目标节点名的映射字典
    # ═══════════════════════════════════════════════════════════

    # Planner → memory_loader（有任务时优先）或 跳过到 strategy
    workflow.add_conditional_edges(
        "planner", route_after_planner, {
            "memory_loader": "memory_loader",
            "match_analysis_agent": "match_analysis_agent",
            "character_recommendation_agent": "character_recommendation_agent",
            "build_agent": "build_agent",
            "rag_agent": "rag_agent",
            "strategy_agent": "strategy_agent",
        },
    )

    # memory_loader → 下一个数据节点
    workflow.add_conditional_edges(
        "memory_loader", route_after_memory, {
            "match_analysis_agent": "match_analysis_agent",
            "character_recommendation_agent": "character_recommendation_agent",
            "build_agent": "build_agent",
            "rag_agent": "rag_agent",
            "strategy_agent": "strategy_agent",
        },
    )

    # match_analysis_agent → 下一个数据节点
    workflow.add_conditional_edges(
        "match_analysis_agent", route_after_match_analysis, {
            "character_recommendation_agent": "character_recommendation_agent",
            "build_agent": "build_agent",
            "rag_agent": "rag_agent",
            "strategy_agent": "strategy_agent",
        },
    )

    # character_recommendation_agent → rag / build / strategy
    workflow.add_conditional_edges(
        "character_recommendation_agent", route_after_character_recommendation, {
            "rag_agent": "rag_agent",
            "build_agent": "build_agent",
            "strategy_agent": "strategy_agent",
        },
    )

    # rag_agent → build / strategy
    # RAG 在 Build 之前执行：攻略可能包含出装信息，先检索再推荐
    workflow.add_conditional_edges(
        "rag_agent", route_after_rag, {
            "build_agent": "build_agent",
            "strategy_agent": "strategy_agent",
        },
    )

    # build_agent → strategy（数据节点层终点）
    workflow.add_conditional_edges(
        "build_agent", route_after_build, {
            "strategy_agent": "strategy_agent",
        },
    )

    # ═══════════════════════════════════════════════════════════
    # 固定边：策略合成 → 回复 → 评估 → 结束
    # 这三个节点始终执行，不受 planned_tasks 影响
    # ═══════════════════════════════════════════════════════════
    workflow.add_edge("strategy_agent", "response_agent")
    workflow.add_edge("response_agent", "evaluation_logger")
    workflow.add_edge("evaluation_logger", END)

    # compile() 将图编译为可执行对象，同时做拓扑排序验证
    return workflow.compile()


# 模块级 graph 实例 —— main.py 和 api.py 直接 import 使用
# LangGraph CLI (langgraph dev) 也通过此变量发现 graph
graph = build_graph()
