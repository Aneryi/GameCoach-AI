"""LangGraph 多 Agent 工作流编排。

从线性流水线升级为条件路由 DAG：
    START → input_normalizer → planner
                                  |
                        [条件路由]
                                  |
              ┌───────────────────┼───────────────────┐
              ↓                   ↓                   ↓
        memory_loader      match_analysis      hero_recommendation
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ↓
                            strategy_agent
                                  │
                            response_agent
                                  │
                          evaluation_logger → END
"""

from langgraph.graph import END, START, StateGraph

from gamecoach.graph.nodes import (
    build_agent,
    evaluation_logger,
    hero_recommendation_agent,
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
    route_after_hero_recommendation,
    route_after_match_analysis,
    route_after_memory,
    route_after_planner,
    route_after_rag,
)
from gamecoach.graph.state import GameCoachState


def build_graph():
    workflow = StateGraph(GameCoachState)

    # ── 注册所有节点 ──
    workflow.add_node("input_normalizer", input_normalizer)
    workflow.add_node("planner", planner)
    workflow.add_node("memory_loader", memory_loader)
    workflow.add_node("match_analysis_agent", match_analysis_agent)
    workflow.add_node("hero_recommendation_agent", hero_recommendation_agent)
    workflow.add_node("build_agent", build_agent)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("strategy_agent", strategy_agent)
    workflow.add_node("response_agent", response_agent)
    workflow.add_node("evaluation_logger", evaluation_logger)

    # ── 固定边 ──
    workflow.add_edge(START, "input_normalizer")
    workflow.add_edge("input_normalizer", "planner")

    # ── 条件边：Planner → Memory / 跳过到 Strategy ──
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "memory_loader": "memory_loader",
            "match_analysis_agent": "match_analysis_agent",
            "hero_recommendation_agent": "hero_recommendation_agent",
            "build_agent": "build_agent",
            "rag_agent": "rag_agent",
            "strategy_agent": "strategy_agent",
        },
    )

    # ── 条件边：Memory → 下一个需要的数据节点 / Strategy ──
    workflow.add_conditional_edges(
        "memory_loader",
        route_after_memory,
        {
            "match_analysis_agent": "match_analysis_agent",
            "hero_recommendation_agent": "hero_recommendation_agent",
            "build_agent": "build_agent",
            "rag_agent": "rag_agent",
            "strategy_agent": "strategy_agent",
        },
    )

    # ── 条件边：Match Analysis → 下一个 / Strategy ──
    workflow.add_conditional_edges(
        "match_analysis_agent",
        route_after_match_analysis,
        {
            "hero_recommendation_agent": "hero_recommendation_agent",
            "build_agent": "build_agent",
            "rag_agent": "rag_agent",
            "strategy_agent": "strategy_agent",
        },
    )

    # ── 条件边：Hero Recommendation → 下一个 / Strategy ──
    workflow.add_conditional_edges(
        "hero_recommendation_agent",
        route_after_hero_recommendation,
        {
            "build_agent": "build_agent",
            "rag_agent": "rag_agent",
            "strategy_agent": "strategy_agent",
        },
    )

    # ── 条件边：Build Agent → 下一个 / Strategy ──
    workflow.add_conditional_edges(
        "build_agent",
        route_after_build,
        {
            "rag_agent": "rag_agent",
            "strategy_agent": "strategy_agent",
        },
    )

    # ── 条件边：RAG → Strategy ──
    workflow.add_conditional_edges(
        "rag_agent",
        route_after_rag,
        {
            "strategy_agent": "strategy_agent",
        },
    )

    # ── 固定边：后处理流水线 ──
    workflow.add_edge("strategy_agent", "response_agent")
    workflow.add_edge("response_agent", "evaluation_logger")
    workflow.add_edge("evaluation_logger", END)

    return workflow.compile()


graph = build_graph()
