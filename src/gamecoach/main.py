"""GameCoach AI CLI entry point."""

from __future__ import annotations

import argparse
import logging

from gamecoach.config.llm import get_chat_model
from gamecoach.graph.workflow import graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEMO_SCENARIOS = [
    {"name": "Rank climbing help", "input": {"user_message": "My win rate has been really low lately and I keep dying in teamfights. How can I climb?", "player_id": "player_001"}},
    {"name": "Character recommendation", "input": {"user_message": "What characters are strong in the current meta for me to learn?", "player_id": "player_001"}},
    {"name": "Build advice", "input": {"user_message": "What's the best build for Alpha?", "player_id": "player_001"}},
    {"name": "Training plan", "input": {"user_message": "Create a 7-day training plan to help me improve.", "player_id": "player_001"}},
]


def _run_and_print(graph_input: dict):
    llm = get_chat_model()
    if llm is None:
        logger.warning("[!] LLM key not configured, using fallback mode.")

    try:
        result = graph.invoke(graph_input)
    except Exception:
        logger.exception("Graph execution failed")
        return

    routing_path = result.get("routing_decisions", {}).get("routing_path", "unknown")
    print(f"\n[Path] {routing_path}")
    print(f"[Tasks] {result['metrics'].get('planner_task_count', 0)}")
    print(f"[RAG hits] {result['metrics'].get('rag_hit_count', 0)}")
    print()
    print(result.get("final_response", "No response generated."))
    metrics = result.get("metrics", {})
    print(f"\n--- Metrics ---")
    print(f"  Path: {metrics.get('routing_path', 'N/A')}")
    print(f"  Degraded: {len(result.get('degraded_nodes', []))}")
    print(f"  Response length: {metrics.get('response_length', 0)} chars")


def main():
    parser = argparse.ArgumentParser(description="GameCoach AI")
    parser.add_argument("--message", "-m", type=str, help="Player question")
    parser.add_argument("--player-id", type=str, default="player_001")
    parser.add_argument("--demo", type=int, default=None, help="Run demo scenario (0-3)")
    parser.add_argument("--all-demos", action="store_true")
    args = parser.parse_args()

    if args.all_demos:
        for i, s in enumerate(DEMO_SCENARIOS):
            print(f"\n{'='*60}\nScenario {i+1}/{len(DEMO_SCENARIOS)}: {s['name']}\n{'='*60}")
            _run_and_print(s["input"])
    elif args.demo is not None:
        if args.demo >= len(DEMO_SCENARIOS):
            logger.error("Demo index %d out of range (0-%d)", args.demo, len(DEMO_SCENARIOS)-1)
            return
        _run_and_print(DEMO_SCENARIOS[args.demo]["input"])
    elif args.message:
        _run_and_print({"user_message": args.message, "player_id": args.player_id})
    else:
        logger.info("No arguments. Usage: python -m gamecoach.main --message 'your question'")
        _run_and_print(DEMO_SCENARIOS[0]["input"])


if __name__ == "__main__":
    main()
