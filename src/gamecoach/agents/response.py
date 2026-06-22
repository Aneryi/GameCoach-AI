"""LLM-Powered Response Agent.

Synthesizes all intermediate agent outputs into a structured player-facing response.
"""

from __future__ import annotations

import logging

from gamecoach.config.llm import get_chat_model
from gamecoach.graph.state import GameCoachState

logger = logging.getLogger(__name__)


def _is_chinese(text: str) -> bool:
    return any('一' <= c <= '鿿' for c in text)


def _fallback_response(state: GameCoachState) -> GameCoachState:
    metrics = state.get("match_analysis", {}).get("metrics", {})
    recommendations = state.get("character_recommendations", [])
    strategy = state.get("strategy", {})
    plan = state.get("training_plan", {})
    rag = state.get("rag_context", [])
    errors = state.get("errors", [])
    degraded = state.get("degraded_nodes", [])
    user_msg = state.get("normalized_message", state.get("user_message", ""))

    cn = _is_chinese(user_msg)
    win_rate = metrics.get("win_rate", 0) if metrics else 0
    avg_kda = metrics.get("avg_kda", 0) if metrics else 0
    avg_deaths = metrics.get("avg_deaths", 0) if metrics else 0
    participation = metrics.get("teamfight_participation", 0) if metrics else 0

    char_lines = ""
    if recommendations:
        char_lines = "\n".join(
            f"- {item.get('character', '?')}: {'; '.join(item.get('fit_reasons', []))}"
            for item in recommendations
        )

    task_lines = ""
    daily_tasks = plan.get("daily_tasks", [])
    if daily_tasks:
        task_lines = "\n".join(
            f"Day {t.get('day', '?')}: {t.get('theme', 'Training')} — "
            f"{t.get('tasks', [''])[0] if t.get('tasks') else ''}"
            for t in daily_tasks
        )

    actions = strategy.get("action_items", ["Improve gameplay based on data."])

    degraded_note = ""
    if degraded:
        if cn:
            degraded_note = "\n[!] 以下模块已跳过: " + ", ".join(degraded) + "\n"
        else:
            degraded_note = "\n[!] Note: Some modules skipped — " + ", ".join(degraded) + "\n"

    rag_note = ""
    if rag:
        prefix = "[参考攻略]" if cn else "[Reference]"
        rag_note = f"\n{prefix}\n" + "\n".join(
            f"- {d.get('title', d.get('source', 'Guide'))}" for d in rag[:3]
        )

    if cn:
        final_response = f"""结论：
{strategy.get('diagnosis', '主要问题是死亡过多，需要降低中期风险。')}

数据依据：
- 最近胜率：{win_rate * 100:.0f}%
- 平均 KDA：{avg_kda}
- 平均死亡：{avg_deaths}
- 参团率：{participation * 100:.0f}%
{degraded_note}
优先改进：
1. {actions[0] if len(actions) > 0 else '改善站位'}
2. {actions[1] if len(actions) > 1 else '固定角色池'}
3. {actions[2] if len(actions) > 2 else '提高参团率'}"""

        if char_lines:
            final_response += f"\n\n推荐角色：\n{char_lines}"
        if task_lines:
            final_response += f"\n\n{plan.get('duration_days', 3)} 天训练计划：\n{task_lines}"
    else:
        final_response = f"""Conclusion:
{strategy.get('diagnosis', 'Your main issue is excessive deaths; reduce mid-game risk.')}

Data:
- Recent win rate: {win_rate * 100:.0f}%
- Avg KDA: {avg_kda}
- Avg deaths: {avg_deaths}
- Teamfight participation: {participation * 100:.0f}%
{degraded_note}
Priority improvements:
1. {actions[0] if len(actions) > 0 else 'Improve positioning'}
2. {actions[1] if len(actions) > 1 else 'Fix character pool'}
3. {actions[2] if len(actions) > 2 else 'Increase teamfight participation'}"""

        if char_lines:
            final_response += f"\n\nRecommended characters:\n{char_lines}"
        if task_lines:
            final_response += f"\n\n{plan.get('duration_days', 3)}-day training plan:\n{task_lines}"

    if rag_note:
        final_response += f"\n{rag_note}"
    if errors:
        final_response += f"\n\n[!] Issues: {'; '.join(errors)}"

    return {"final_response": final_response}


def create_llm_response(state: GameCoachState) -> GameCoachState:
    llm = get_chat_model(temperature=0.4)
    if llm is None:
        logger.info("LLM unavailable, using fallback response.")
        return _fallback_response(state)

    analysis = state.get("match_analysis", {})
    metrics = analysis.get("metrics", {})
    memory = state.get("memory", {})
    recommendations = state.get("character_recommendations", [])
    strategy = state.get("strategy", {})
    plan = state.get("training_plan", {})
    rag = state.get("rag_context", [])
    degraded = state.get("degraded_nodes", [])
    user_msg = state.get("normalized_message", state.get("user_message", ""))

    context_parts = [f"Player question: {user_msg}"]
    if metrics:
        context_parts.append(
            f"Stats: {metrics.get('win_rate', 0):.0%} win rate, KDA {metrics.get('avg_kda', 0)}, "
            f"{metrics.get('avg_deaths', 0)} avg deaths, {metrics.get('teamfight_participation', 0):.0%} participation"
        )
    if memory.get("rank"):
        context_parts.append(f"Rank: {memory['rank']}, roles: {', '.join(memory.get('main_roles', []))}")
    if recommendations:
        rec_text = "Recommended: " + ", ".join(
            f"{r.get('character', '?')} ({'; '.join(r.get('fit_reasons', []))})" for r in recommendations[:3]
        )
        context_parts.append(rec_text)
    if strategy:
        context_parts.append(f"Diagnosis: {strategy.get('diagnosis', '')}")
        context_parts.append(f"Priorities: {' > '.join(strategy.get('priorities', []))}")
    if plan.get("daily_tasks"):
        plan_text = f"{plan.get('duration_days', '?')}-day plan: " + "; ".join(
            f"Day{t.get('day', '?')}: {t.get('theme', '')}" for t in plan.get("daily_tasks", [])
        )
        context_parts.append(plan_text)
    if rag:
        rag_text = "Guides: " + "; ".join(
            f"{d.get('title', '')}: {d.get('snippet', '')}" for d in rag[:2]
        )
        context_parts.append(rag_text)
    if degraded:
        context_parts.append(f"Note: Modules skipped — {', '.join(degraded)}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a game coach. Write a clear, actionable response for the player.

{context}

Requirements:
1. Start with a one-sentence conclusion identifying the core issue
2. Cite specific data (win rate, KDA, deaths) as evidence
3. Give 3 priority action items, each specific to a gameplay scenario
4. If characters are recommended, state why
5. If a training plan exists, summarize daily focus
6. If guides are referenced, cite sources
7. Coach tone: direct but encouraging, practical
8. If data is missing, be honest without fabricating
9. Keep under 500 words
10. Respond in the same language as the player's question"""

    try:
        result = llm.invoke(prompt)
        final = result.content if hasattr(result, "content") else str(result)
        return {"final_response": final.strip()}
    except Exception:
        logger.exception("LLM Response failed, falling back")
        return _fallback_response(state)
