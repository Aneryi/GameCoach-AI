"""GameCoach AI 主入口。

支持 CLI 直接运行和参数配置。
无 LLM Key 时自动使用 fallback 模式。
"""

from __future__ import annotations

import argparse
import logging

from gamecoach.config.llm import get_chat_model
from gamecoach.graph.workflow import graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── 预设演示场景 ──

DEMO_SCENARIOS = [
    {
        "name": "上分求助",
        "input": {
            "user_message": "我最近胜率很低，玩射手总是团战暴毙，怎么上分？",
            "player_id": "player_001",
            "game": "moba",
        },
    },
    {
        "name": "英雄推荐",
        "input": {
            "user_message": "当前版本有什么强势英雄适合我练？",
            "player_id": "player_001",
            "game": "moba",
        },
    },
    {
        "name": "出装建议",
        "input": {
            "user_message": "狄仁杰出什么装备比较好？",
            "player_id": "player_001",
            "game": "moba",
        },
    },
    {
        "name": "训练计划",
        "input": {
            "user_message": "帮我制定一个 7 天上分训练计划。",
            "player_id": "player_001",
            "game": "moba",
        },
    },
]


def run_demo(scenario_index: int = 0):
    """运行单个演示场景。"""
    if scenario_index >= len(DEMO_SCENARIOS):
        logger.error("场景序号 %d 超出范围 (0-%d)", scenario_index, len(DEMO_SCENARIOS) - 1)
        return

    scenario = DEMO_SCENARIOS[scenario_index]
    logger.info("运行场景: %s", scenario["name"])
    _run_and_print(scenario["input"])


def run_all_demos():
    """运行所有演示场景。"""
    for i, scenario in enumerate(DEMO_SCENARIOS):
        print(f"\n{'=' * 60}")
        print(f"场景 {i + 1}/{len(DEMO_SCENARIOS)}: {scenario['name']}")
        print(f"{'=' * 60}")
        _run_and_print(scenario["input"])


def run_custom(user_message: str, player_id: str = "player_001", game: str = "moba"):
    """运行自定义输入。"""
    _run_and_print(
        {
            "user_message": user_message,
            "player_id": player_id,
            "game": game,
        }
    )


def _run_and_print(graph_input: dict):
    """执行 Graph 并打印结果。"""
    # 检查 LLM 状态
    llm = get_chat_model()
    if llm is None:
        logger.warning("[!] LLM Key 未配置，将使用 fallback 规则模式。")

    try:
        result = graph.invoke(graph_input)
    except Exception:
        logger.exception("Graph 执行失败")
        return

    routing_path = result.get("routing_decisions", {}).get("routing_path", "未知")
    task_count = result["metrics"].get("planner_task_count", 0)
    rag_hits = result["metrics"].get("rag_hit_count", 0)

    print(f"\n[执行路径] {routing_path}")
    print(f"[任务数] {task_count}")
    print(f"[RAG 命中] {rag_hits} 条")
    print()

    print(result.get("final_response", "未能生成回复。"))

    # 打印评估摘要
    metrics = result.get("metrics", {})
    print(f"\n--- 评估指标 ---")
    print(f"  路由路径: {metrics.get('routing_path', 'N/A')}")
    print(f"  降级节点: {len(result.get('degraded_nodes', []))}")
    print(f"  回复长度: {metrics.get('response_length', 0)} 字符")


def main():
    parser = argparse.ArgumentParser(description="GameCoach AI - 游戏成长教练")
    parser.add_argument(
        "--message", "-m", type=str, help="用户问题（中文）"
    )
    parser.add_argument(
        "--player-id", type=str, default="player_001", help="玩家 ID"
    )
    parser.add_argument(
        "--game", type=str, default="moba", help="游戏类型"
    )
    parser.add_argument(
        "--demo", type=int, default=None, help="运行预设演示场景 (0-3)"
    )
    parser.add_argument(
        "--all-demos", action="store_true", help="运行全部演示场景"
    )

    args = parser.parse_args()

    if args.all_demos:
        run_all_demos()
    elif args.demo is not None:
        run_demo(args.demo)
    elif args.message:
        run_custom(args.message, args.player_id, args.game)
    else:
        # 默认运行第一个演示
        logger.info("未指定参数，运行默认演示场景。")
        logger.info("用法: python -m gamecoach.main --message '你的问题'")
        logger.info("用法: python -m gamecoach.main --demo 0")
        logger.info("用法: python -m gamecoach.main --all-demos\n")
        run_demo(0)


if __name__ == "__main__":
    main()
