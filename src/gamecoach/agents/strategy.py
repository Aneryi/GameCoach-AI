"""LLM-Powered Strategy Agent.

Synthesizes match analysis, player profile, character recommendations,
and RAG results into structured strategy advice and training plans.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field, ValidationError

from gamecoach.config.llm import get_chat_model
from gamecoach.graph.state import GameCoachState

logger = logging.getLogger(__name__)


class DailyTask(BaseModel):
    day: int
    theme: str
    tasks: list[str]
    success_criteria: list[str]


class TrainingPlan(BaseModel):
    duration_days: int = Field(description="3, 7, or 14")
    goal: str
    daily_tasks: list[DailyTask]
    review_checkpoints: list[str]


class StrategyOutput(BaseModel):
    diagnosis: str
    priorities: list[str]
    action_items: list[str]
    avoid_items: list[str]
    training_plan: TrainingPlan


def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0).strip()
    return text.strip()


def _has_chinese(text: str) -> bool:
    return any('一' <= c <= '鿿' for c in text)


def _fallback_strategy(state: GameCoachState) -> GameCoachState:
    analysis = state.get("match_analysis", {})
    memory = state.get("memory", {})
    weaknesses = analysis.get("weaknesses", []) + memory.get("weaknesses", [])
    user_msg = state.get("normalized_message", state.get("user_message", ""))
    cn = _has_chinese(user_msg)

    training_plan = {
        "duration_days": 3,
        "goal": "先把胜率拉回 50% 以上" if cn else "Stabilize win rate above 50%",
        "daily_tasks": [
            {"day": 1, "theme": "减少无效死亡" if cn else "Reduce unnecessary deaths",
             "tasks": ["每次死亡后记录前 3 次原因" if cn else "Record first 3 death causes each game",
                       "10 分钟后没有视野不单带过河道" if cn else "Avoid pushing without vision after 10 min"],
             "success_criteria": ["单局死亡 <= 5" if cn else "Deaths <= 5 per game"]},
            {"day": 2, "theme": "团战站位" if cn else "Teamfight positioning",
             "tasks": ["保持在前排后方输出" if cn else "Stay behind frontline",
                       "等敌方关键技能交出后再进场" if cn else "Wait for enemy engage before committing"],
             "success_criteria": ["参团率 >= 55%" if cn else "Participation >= 55%",
                                  "团战先手死亡次数为 0" if cn else "Zero first-deaths in teamfights"]},
            {"day": 3, "theme": "固定角色池" if cn else "Narrow character pool",
             "tasks": ["只使用 2 个推荐角色排位" if cn else "Only use 2 recommended characters",
                       "复盘胜负局的经济差和死亡点" if cn else "Review economy and death points after each game"],
             "success_criteria": ["连续 3 局使用同一套复盘模板" if cn else "3 consecutive games on same characters"]},
        ],
        "review_checkpoints": ["第 3 天复盘 KDA" if cn else "Day 3: review KDA",
                               "对比本周胜率变化" if cn else "Compare win rate change"],
    }
    strategy = {
        "diagnosis": "当前优先级不是扩角色池，而是降低中期掉点和团战暴毙。" if cn
        else "Priority is not expanding character pool, but reducing mid-game deaths and teamfight mistakes.",
        "priorities": ["少死" if cn else "Die less",
                       "固定角色池" if cn else "Fix character pool",
                       "提高中期参团率" if cn else "Improve teamfight participation"],
        "weaknesses": weaknesses,
        "action_items": [
            "10 分钟后只吃安全线，队友不在附近时不压深线。" if cn
            else "Only farm safe lanes after 10 min; don't push deep without teammates nearby.",
            "团战保持在前排后方输出，敌方突进未露头前不交位移向前。" if cn
            else "Stay behind frontline in teamfights; don't use mobility forward until enemy dive is revealed.",
            "先用 2 个容错较高角色打 20 场，避免频繁换角色导致复盘失效。" if cn
            else "Play 20 games on 2 comfortable characters before switching.",
        ],
    }
    return {"strategy": strategy, "training_plan": training_plan}


def create_llm_strategy(state: GameCoachState) -> GameCoachState:
    llm = get_chat_model(temperature=0.3)
    if llm is None:
        logger.info("LLM unavailable, using fallback strategy.")
        return _fallback_strategy(state)

    analysis = state.get("match_analysis", {})
    memory = state.get("memory", {})
    recommendations = state.get("character_recommendations", [])
    rag_context = state.get("rag_context", [])
    user_msg = state.get("normalized_message", state.get("user_message", ""))

    metrics = analysis.get("metrics", {})
    analysis_weaknesses = analysis.get("weaknesses", [])
    memory_weaknesses = memory.get("weaknesses", [])
    strengths = analysis.get("strengths", [])
    win_rate = metrics.get("win_rate", 0)
    avg_kda = metrics.get("avg_kda", 0)
    avg_deaths = metrics.get("avg_deaths", 0)
    participation = metrics.get("teamfight_participation", 0)

    char_text = ""
    if recommendations:
        char_text = "\nRecommended characters:"
        for r in recommendations[:3]:
            char_text += f"\n- {r.get('character', '?')}: {'; '.join(r.get('fit_reasons', []))}"

    rag_text = ""
    if rag_context:
        snippets = [d.get("snippet", "") for d in rag_context[:3] if d.get("snippet")]
        if snippets:
            rag_text = "\nRelevant guides:\n" + "\n".join(f"- {s}" for s in snippets)

    prompt = f"""You are a game coaching strategy generator. Output JSON strategy advice and training plan.

Player question: {user_msg}

Data:
- Win rate: {win_rate:.0%}
- Avg KDA: {avg_kda}
- Avg deaths: {avg_deaths}
- Teamfight participation: {participation:.0%}
- Identified weaknesses: {', '.join(analysis_weaknesses) if analysis_weaknesses else 'none'}
- Player weaknesses: {', '.join(memory_weaknesses) if memory_weaknesses else 'none'}
- Strengths: {', '.join(strengths) if strengths else 'none'}
{char_text}
{rag_text}

Output ONLY this JSON:
{{
  "diagnosis": "Core problem diagnosis (2-3 sentences)",
  "priorities": ["Priority 1", "Priority 2", "Priority 3"],
  "action_items": ["Actionable item 1", "item 2", "item 3"],
  "avoid_items": ["Behavior to avoid 1", "avoid 2"],
  "training_plan": {{
    "duration_days": 3,
    "goal": "Training goal",
    "daily_tasks": [
      {{"day": 1, "theme": "Theme", "tasks": ["Task 1", "Task 2"], "success_criteria": ["Criteria"]}}
    ],
    "review_checkpoints": ["Checkpoint 1", "Checkpoint 2"]
  }}
}}

Rules:
- duration_days: 3 for simple, 7 for moderate, 14 for complex issues
- Daily themes should not repeat
- Respond in the same language as the user's question"""

    try:
        result = llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        data = json.loads(_extract_json(text))
        validated = StrategyOutput(**data)
        logger.info("LLM Strategy generated %d-day plan", validated.training_plan.duration_days)
        return {
            "strategy": {
                "diagnosis": validated.diagnosis,
                "priorities": validated.priorities,
                "weaknesses": analysis_weaknesses + memory_weaknesses,
                "action_items": validated.action_items,
                "avoid_items": validated.avoid_items,
            },
            "training_plan": validated.training_plan.model_dump(),
        }
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        logger.warning("LLM Strategy parse failed: %s, falling back", e)
        return _fallback_strategy(state)
    except Exception:
        logger.exception("LLM Strategy failed, falling back")
        return _fallback_strategy(state)
