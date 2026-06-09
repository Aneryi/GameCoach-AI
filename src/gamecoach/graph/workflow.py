from langgraph.graph import END, START, StateGraph

from gamecoach.graph.nodes import (
    evaluation_logger,
    hero_recommendation_agent,
    input_normalizer,
    match_analysis_agent,
    memory_loader,
    planner,
    response_agent,
    strategy_agent,
)
from gamecoach.graph.state import GameCoachState


def build_graph():
    workflow = StateGraph(GameCoachState)

    workflow.add_node("input_normalizer", input_normalizer)
    workflow.add_node("planner", planner)
    workflow.add_node("memory_loader", memory_loader)
    workflow.add_node("match_analysis_agent", match_analysis_agent)
    workflow.add_node("hero_recommendation_agent", hero_recommendation_agent)
    workflow.add_node("strategy_agent", strategy_agent)
    workflow.add_node("response_agent", response_agent)
    workflow.add_node("evaluation_logger", evaluation_logger)

    workflow.add_edge(START, "input_normalizer")
    workflow.add_edge("input_normalizer", "planner")
    workflow.add_edge("planner", "memory_loader")
    workflow.add_edge("memory_loader", "match_analysis_agent")
    workflow.add_edge("match_analysis_agent", "hero_recommendation_agent")
    workflow.add_edge("hero_recommendation_agent", "strategy_agent")
    workflow.add_edge("strategy_agent", "response_agent")
    workflow.add_edge("response_agent", "evaluation_logger")
    workflow.add_edge("evaluation_logger", END)

    return workflow.compile()


graph = build_graph()

