"""LangGraph multi-agent workflow with conditional routing.

Execution DAG:
    START → input_normalizer → planner
                                  |
                        [conditional routing]
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
                          strategy_agent
                                  |
                          response_agent
                                  |
                        evaluation_logger → END
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
    workflow = StateGraph(GameCoachState)

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

    workflow.add_edge(START, "input_normalizer")
    workflow.add_edge("input_normalizer", "planner")

    workflow.add_conditional_edges("planner", route_after_planner, {
        "memory_loader": "memory_loader",
        "match_analysis_agent": "match_analysis_agent",
        "character_recommendation_agent": "character_recommendation_agent",
        "build_agent": "build_agent",
        "rag_agent": "rag_agent",
        "strategy_agent": "strategy_agent",
    })

    workflow.add_conditional_edges("memory_loader", route_after_memory, {
        "match_analysis_agent": "match_analysis_agent",
        "character_recommendation_agent": "character_recommendation_agent",
        "build_agent": "build_agent",
        "rag_agent": "rag_agent",
        "strategy_agent": "strategy_agent",
    })

    workflow.add_conditional_edges("match_analysis_agent", route_after_match_analysis, {
        "character_recommendation_agent": "character_recommendation_agent",
        "build_agent": "build_agent",
        "rag_agent": "rag_agent",
        "strategy_agent": "strategy_agent",
    })

    workflow.add_conditional_edges("character_recommendation_agent", route_after_character_recommendation, {
        "rag_agent": "rag_agent",
        "build_agent": "build_agent",
        "strategy_agent": "strategy_agent",
    })

    workflow.add_conditional_edges("rag_agent", route_after_rag, {
        "build_agent": "build_agent",
        "strategy_agent": "strategy_agent",
    })

    workflow.add_conditional_edges("build_agent", route_after_build, {
        "strategy_agent": "strategy_agent",
    })

    workflow.add_edge("strategy_agent", "response_agent")
    workflow.add_edge("response_agent", "evaluation_logger")
    workflow.add_edge("evaluation_logger", END)

    return workflow.compile()


graph = build_graph()
