"""
GameCoach AI CLI 入口。

支持通过命令行参数运行教练服务，提供预设演示场景和自定义输入。
"""

from __future__ import annotations

import argparse
import logging

from gamecoach.config.llm import get_chat_model
from gamecoach.graph.workflow import graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── 预设演示场景 ──
# 每个场景包含玩家输入和玩家 ID，用于快速演示不同能力

DEMO_SCENARIOS = [
    {
        "name": "上分求助",
        "input": {
            "user_message": "我最近胜率很低，怎么上分？",
            "player_id": "player_001",
        },
    },
    {
        "name": "角色推荐",
        "input": {
            "user_message": "当前版本有什么强势角色适合我练？",
            "player_id": "player_001",
        },
    },
    {
        "name": "出装建议",
        "input": {
            "user_message": "Alpha 出什么装备比较好？",
            "player_id": "player_001",
        },
    },
    {
        "name": "训练计划",
        "input": {
            "user_message": "帮我制定一个 7 天上分训练计划。",
            "player_id": "player_001",
        },
    },
]


def _run_and_print(graph_input: dict):
    """执行 Graph 并打印结果到终端。"""
    # 检查 LLM 状态（仅提示，不影响执行——节点内部有 fallback）
    llm = get_chat_model()
    if llm is None:
        logger.warning("[!] LLM key 未配置，使用 fallback 模式。")

    try:
        result = graph.invoke(graph_input)
    except Exception:
        logger.exception("Graph 执行失败")
        return

    routing_path = result.get("routing_decisions", {})
    # routing_path 是一个 dict，用节点名作为 key。取实际执行的路径。
    executed = [k for k, v in routing_path.items() if v in ("execute", "fixed")]
    path_str = " -> ".join(executed) if executed else "unknown"

    print(f"\n[Path] {path_str}")
    print(f"[Tasks] {result['metrics'].get('planner_task_count', 0)}")
    print(f"[RAG hits] {result['metrics'].get('rag_hit_count', 0)}")
    print()
    print(result.get("final_response", "未能生成回复。"))

    metrics = result.get("metrics", {})
    print(f"\n--- Metrics ---")
    print(f"  Path: {metrics.get('routing_path', 'N/A')}")
    print(f"  Degraded: {len(result.get('degraded_nodes', []))}")
    print(f"  Response length: {metrics.get('response_length', 0)} chars")


def main():
    parser = argparse.ArgumentParser(description="GameCoach AI — 游戏成长教练")
    parser.add_argument("--message", "-m", type=str, help="玩家问题（中文或英文）")
    parser.add_argument("--player-id", type=str, default="player_001", help="玩家 ID")
    parser.add_argument("--demo", type=int, default=None, help="运行预设演示场景 (0-3)")
    parser.add_argument("--all-demos", action="store_true", help="运行全部演示场景")

    args = parser.parse_args()

    if args.all_demos:
        for i, s in enumerate(DEMO_SCENARIOS):
            print(f"\n{'='*60}")
            print(f"场景 {i+1}/{len(DEMO_SCENARIOS)}: {s['name']}")
            print(f"{'='*60}")
            _run_and_print(s["input"])
    elif args.demo is not None:
        if args.demo >= len(DEMO_SCENARIOS):
            logger.error("Demo index %d out of range (0-%d)", args.demo, len(DEMO_SCENARIOS)-1)
            return
        _run_and_print(DEMO_SCENARIOS[args.demo]["input"])
    elif args.message:
        _run_and_print({"user_message": args.message, "player_id": args.player_id})
    else:
        # 默认：运行第一个演示场景
        logger.info("未指定参数，运行默认演示。")
        logger.info("用法: python -m gamecoach.main --message '你的问题'")
        logger.info("用法: python -m gamecoach.main --demo 0")
        logger.info("用法: python -m gamecoach.main --all-demos\n")
        _run_and_print(DEMO_SCENARIOS[0]["input"])


if __name__ == "__main__":
    main()
