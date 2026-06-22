"""
LLM-Powered Response Agent。

汇总所有中间 Agent 的输出，生成面向玩家的结构化最终回复。
与 Strategy Agent 的区别：Strategy 输出结构化 JSON（给系统消费），
Response 输出自然语言文本（给玩家阅读）。

输入：state 中的所有中间结果（match_analysis, character_recommendations,
      strategy, training_plan, rag_context, degraded_nodes）
输出：final_response（自然语言字符串）
"""

from __future__ import annotations

import logging

from gamecoach.config.llm import get_chat_model
from gamecoach.graph.state import GameCoachState

logger = logging.getLogger(__name__)


def _is_chinese(text: str) -> bool:
    """检测文本是否含中文字符（Unicode CJK 范围）。"""
    return any('一' <= c <= '鿿' for c in text)


# ── Fallback Response ──

def _fallback_response(state: GameCoachState) -> GameCoachState:
    """
    LLM 不可用时的模板版回复合成器。

    使用 f-string 拼装各模块的输出结果。
    自动检测用户输入语言（中文/英文），选择对应语言的模板。
    """
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
{strategy.get('diagnosis', 'Your main issue is excessive deaths.')}

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


# ── LLM Response ──

def create_llm_response(state: GameCoachState) -> GameCoachState:
    """
    LLM 驱动的回复合成。

    将所有中间结果整理为结构化上下文，通过 prompt 指示 LLM
    生成玩家友好的 Coaching 回复。

    回复结构（prompt 中约束）：
    1. 一句话结论 → 核心问题
    2. 数据依据 → 胜率/KDA/死亡等具体数字
    3. 3 条优先改进动作 → 每条具体到场景
    4. 角色推荐（如有）→ 附推荐理由
    5. 训练计划（如有）→ 每日重点概述
    6. 攻略引用（如有）→ 注明来源
    """
    llm = get_chat_model(temperature=0.4)  # 稍高温：回复需要自然流畅
    if llm is None:
        logger.info("LLM 不可用，使用 fallback response。")
        return _fallback_response(state)

    # ── 收集所有中间结果 ──
    analysis = state.get("match_analysis", {})
    metrics = analysis.get("metrics", {})
    memory = state.get("memory", {})
    recommendations = state.get("character_recommendations", [])
    strategy = state.get("strategy", {})
    plan = state.get("training_plan", {})
    rag = state.get("rag_context", [])
    degraded = state.get("degraded_nodes", [])
    user_msg = state.get("normalized_message", state.get("user_message", ""))

    # ── 构建结构化上下文 ──
    context_parts = [f"玩家问题：{user_msg}"]

    if metrics:
        context_parts.append(
            f"战绩：胜率{metrics.get('win_rate', 0):.0%}，KDA {metrics.get('avg_kda', 0)}，"
            f"场均死亡 {metrics.get('avg_deaths', 0)}，参团率 {metrics.get('teamfight_participation', 0):.0%}"
        )
    if memory.get("rank"):
        context_parts.append(
            f"段位 {memory['rank']}，主玩 {'、'.join(memory.get('main_roles', []))}"
        )
    if recommendations:
        rec_text = "推荐角色：" + "、".join(
            f"{r.get('character', '?')}({'; '.join(r.get('fit_reasons', []))})"
            for r in recommendations[:3]
        )
        context_parts.append(rec_text)
    if strategy:
        context_parts.append(f"诊断：{strategy.get('diagnosis', '')}")
        context_parts.append(f"改进优先级：{' > '.join(strategy.get('priorities', []))}")
    if plan.get("daily_tasks"):
        plan_text = f"{plan.get('duration_days', '?')}天训练计划：" + "；".join(
            f"Day{t.get('day', '?')}-{t.get('theme', '')}"
            for t in plan.get("daily_tasks", [])
        )
        context_parts.append(plan_text)
    if rag:
        rag_text = "攻略参考：" + "；".join(
            f"{d.get('title', '')}: {d.get('snippet', '')}" for d in rag[:2]
        )
        context_parts.append(rag_text)
    if degraded:
        context_parts.append(f"以下模块已跳过：{', '.join(degraded)}")

    context = "\n\n".join(context_parts)

    # ── LLM 调用 ──
    prompt = f"""你是游戏成长教练。根据以下分析数据，给玩家一段清晰、可执行、有激励性的回复。

{context}

要求：
1. 开头一句话结论，让玩家知道核心问题是什么
2. 引用具体数据作为依据
3. 给出 3 条最优先的改进动作，每条要具体到场景
4. 如果有推荐角色，说明推荐理由
5. 如果有训练计划，概述每日重点
6. 如果有攻略引用，注明来源
7. 语气像教练 — 直接但不贬低，务实但有激励性
8. 如果某些数据缺失，诚实说明而不编造
9. 不超过 500 字
10. 用和用户输入相同的语言回复

请直接输出给玩家看的回复内容。"""

    try:
        result = llm.invoke(prompt)
        final = result.content if hasattr(result, "content") else str(result)
        return {"final_response": final.strip()}
    except Exception:
        logger.exception("LLM Response 调用失败，回退到模板版")
        return _fallback_response(state)
