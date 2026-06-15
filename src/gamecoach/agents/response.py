"""LLM-Powered Response Agent。

汇总所有中间 Agent 的输出，生成面向玩家的结构化最终回复。
包含结论、数据依据、改进建议、英雄推荐和训练计划。
"""

from __future__ import annotations

import logging

from gamecoach.config.llm import get_chat_model
from gamecoach.graph.state import GameCoachState

logger = logging.getLogger(__name__)


def _fallback_response(state: GameCoachState) -> GameCoachState:
    """LLM 不可用时的模板版 Response。

    使用字符串模板拼装各模块的输出结果。
    """
    metrics = state.get("match_analysis", {}).get("metrics", {})
    recommendations = state.get("hero_recommendations", [])
    strategy = state.get("strategy", {})
    plan = state.get("training_plan", {})
    rag = state.get("rag_context", [])
    errors = state.get("errors", [])
    degraded = state.get("degraded_nodes", [])

    win_rate = metrics.get("win_rate", 0) if metrics else 0
    avg_kda = metrics.get("avg_kda", 0) if metrics else 0
    avg_deaths = metrics.get("avg_deaths", 0) if metrics else 0
    participation = metrics.get("teamfight_participation", 0) if metrics else 0

    hero_lines = ""
    if recommendations:
        hero_lines = "\n".join(
            f"- {item.get('hero', '?')}：{'; '.join(item.get('fit_reasons', []))}"
            for item in recommendations
        )

    task_lines = ""
    daily_tasks = plan.get("daily_tasks", [])
    if daily_tasks:
        task_lines = "\n".join(
            f"Day {t.get('day', '?')}：{t.get('theme', '训练')}，{t.get('tasks', [''])[0] if t.get('tasks') else ''}"
            for t in daily_tasks
        )

    actions = strategy.get("action_items", ["请根据数据改善游戏表现。"])

    # 降级提示
    degraded_note = ""
    if degraded:
        degraded_note = (
            "\n[!] 注意：以下模块因数据/服务不可用已跳过 — "
            + "、".join(degraded)
            + "\n"
        )

    # RAG 引用
    rag_note = ""
    if rag:
        rag_note = "\n[参考攻略]\n" + "\n".join(
            f"- {d.get('title', d.get('source', '攻略'))}" for d in rag[:3]
        )

    final_response = f"""结论：
{strategy.get('diagnosis', '你的主要问题是死亡偏高，需要降低中期掉点风险。')}

数据依据：
- 最近 20 场胜率：{win_rate * 100:.0f}%
- 平均 KDA：{avg_kda}
- 平均死亡：{avg_deaths}
- 平均参团率：{participation * 100:.0f}%
{degraded_note}
优先改进：
1. {actions[0] if len(actions) > 0 else '优化团战站位'}
2. {actions[1] if len(actions) > 1 else '固定英雄池'}
3. {actions[2] if len(actions) > 2 else '提高参团率'}"""

    if hero_lines:
        final_response += f"\n\n推荐英雄：\n{hero_lines}"

    if task_lines:
        final_response += f"\n\n{plan.get('duration_days', 3)} 天训练计划：\n{task_lines}"

    if rag_note:
        final_response += f"\n{rag_note}"

    if errors:
        final_response += f"\n\n[!] 执行中遇到以下问题：{'；'.join(errors)}"

    return {"final_response": final_response}


def create_llm_response(state: GameCoachState) -> GameCoachState:
    """LLM 驱动的回复合成。

    将各 Agent 输出整合为玩家友好的结构化建议。
    """
    llm = get_chat_model(temperature=0.4)
    if llm is None:
        logger.info("LLM 不可用，使用 fallback response。")
        return _fallback_response(state)

    # 收集上下文
    analysis = state.get("match_analysis", {})
    metrics = analysis.get("metrics", {})
    memory = state.get("memory", {})
    recommendations = state.get("hero_recommendations", [])
    strategy = state.get("strategy", {})
    plan = state.get("training_plan", {})
    rag = state.get("rag_context", [])
    degraded = state.get("degraded_nodes", [])
    user_msg = state.get("normalized_message", state.get("user_message", ""))

    # 构建结构化上下文
    context_parts = [f"玩家问题：{user_msg}"]

    if metrics:
        context_parts.append(
            f"战绩：胜率{metrics.get('win_rate', 0):.0%}，KDA {metrics.get('avg_kda', 0)}，"
            f"场均死亡 {metrics.get('avg_deaths', 0)}，参团率 {metrics.get('teamfight_participation', 0):.0%}"
        )

    if memory.get("rank"):
        context_parts.append(
            f"玩家段位{memory['rank']}，主玩{'、'.join(memory.get('main_roles', []))}"
        )

    if recommendations:
        rec_text = "推荐英雄：" + "、".join(
            f"{r.get('hero', '?')}({'; '.join(r.get('fit_reasons', []))})" for r in recommendations[:3]
        )
        context_parts.append(rec_text)

    if strategy:
        context_parts.append(f"诊断：{strategy.get('diagnosis', '')}")
        context_parts.append(f"改进优先级：{' > '.join(strategy.get('priorities', []))}")

    if plan.get("daily_tasks"):
        plan_text = f"{plan.get('duration_days', '?')}天训练计划：" + "；".join(
            f"Day{t.get('day', '?')}-{t.get('theme', '')}" for t in plan.get("daily_tasks", [])
        )
        context_parts.append(plan_text)

    if rag:
        rag_text = "攻略参考：" + "；".join(
            f"{d.get('title', '')}: {d.get('snippet', '')}" for d in rag[:2]
        )
        context_parts.append(rag_text)

    if degraded:
        context_parts.append(f"注意：以下模块已跳过 — {', '.join(degraded)}")

    context = "\n\n".join(context_parts)

    prompt = f"""你是游戏成长教练的回复生成器。根据以下分析数据，给玩家一段清晰、可执行、有激励性的回复。

{context}

要求：
1. 开头一句话结论，让玩家知道核心问题是什么
2. 引用具体数据（胜率、KDA、死亡等）作为依据
3. 给出 3 条最优先的改进动作，每条要具体到场景
4. 如果有推荐英雄，说明推荐理由
5. 如果有训练计划，概述每日重点
6. 如果有攻略引用，注明来源
7. 语气像教练 — 直接但不贬低，务实但有激励性
8. 如果某些数据缺失，诚实说明而不编造
9. 用中文，分段清晰，不超过 500 字

请直接输出给玩家看的回复内容。"""

    try:
        result = llm.invoke(prompt)
        final = result.content if hasattr(result, "content") else str(result)
        return {"final_response": final.strip()}
    except Exception:
        logger.exception("LLM Response 调用失败，回退到模板版")
        return _fallback_response(state)
