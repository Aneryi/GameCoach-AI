"""LLM-Powered Planner Agent.

Uses LLM to decompose natural language into structured task lists.
Falls back to keyword matching when LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field, ValidationError

from gamecoach.config.llm import get_chat_model
from gamecoach.graph.state import GameCoachState

logger = logging.getLogger(__name__)


class PlannedTaskModel(BaseModel):
    task_id: str = Field(description="Task ID, e.g. t1, t2")
    task_type: str = Field(
        description=(
            "Task type: match_analysis, character_recommendation, "
            "build_recommendation, strategy_generation, "
            "rag_lookup, memory_lookup, training_plan"
        )
    )
    description: str = Field(description="One-line task description")
    priority: int = Field(description="Priority, 1 = highest")
    required_tools: list[str] = Field(
        description="Required tools: match_history_tool / character_database_tool / patch_meta_tool / guide_rag_tool"
    )


class PlannedTaskList(BaseModel):
    intent: str = Field(description="User intent classification")
    planned_tasks: list[PlannedTaskModel] = Field(description="Task breakdown")


def _fallback_planner(state: GameCoachState) -> GameCoachState:
    message = state.get("normalized_message", state.get("user_message", ""))
    tasks = [
        PlannedTaskModel(task_id="t1", task_type="memory_lookup", description="Load player profile", priority=1, required_tools=[]),
        PlannedTaskModel(task_id="t2", task_type="match_analysis", description="Analyze recent match performance", priority=2, required_tools=["match_history_tool"]),
        PlannedTaskModel(task_id="t3", task_type="character_recommendation", description="Recommend characters based on meta and preferences", priority=3, required_tools=["character_database_tool", "patch_meta_tool"]),
        PlannedTaskModel(task_id="t4", task_type="strategy_generation", description="Generate strategy advice and training focus", priority=4, required_tools=[]),
    ]
    intent = "improve_performance"
    if any(w in message for w in ["character", "play", "练什么", "英雄"]):
        intent = "character_recommendation"
    if any(w in message for w in ["build", "item", "出装", "装备"]):
        intent = "build_recommendation"
    if any(w in message for w in ["guide", "strategy", "攻略", "怎么打"]):
        intent = "strategy_help"
    return {"intent": intent, "planned_tasks": [t.model_dump() for t in tasks]}


def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0).strip()
    return text.strip()


def create_llm_planner(state: GameCoachState) -> GameCoachState:
    message = state.get("normalized_message", state.get("user_message", ""))
    memory = state.get("memory", {})

    llm = get_chat_model(temperature=0.1)
    if llm is None:
        logger.info("LLM unavailable, using fallback planner.")
        return _fallback_planner(state)

    memory_context = ""
    if memory:
        rank = memory.get("rank", "unknown")
        roles = ", ".join(memory.get("main_roles", []))
        goals = ", ".join(memory.get("goals", []))
        memory_context = f"\nPlayer profile: rank {rank}, roles {roles}, goals {goals}."

    prompt = f"""You are a game coaching task planner. Analyze the player's input and output a JSON task breakdown.

Player input: {message}{memory_context}

Available task types:
- match_analysis: Analyze match data (win rate, KDA, deaths, etc.)
- character_recommendation: Recommend characters for climbing
- build_recommendation: Recommend equipment/builds
- strategy_generation: Generate strategy and gameplay advice
- rag_lookup: Search guides for relevant tips
- memory_lookup: Load player profile
- training_plan: Generate a phased training plan

Rules:
1. Most questions need memory_lookup first
2. Match/performance questions need match_analysis
3. Character selection questions need character_recommendation
4. Equipment/build questions need build_recommendation
5. Strategy/gameplay questions need rag_lookup and strategy_generation
6. Training plan requests need training_plan
7. Each task_type appears at most once
8. Not all tasks are needed for every question
9. Respond in the same language as the user's input

Output ONLY this JSON format:
{{
  "intent": "improve_performance",
  "planned_tasks": [
    {{"task_id": "t1", "task_type": "memory_lookup", "description": "Load player profile", "priority": 1, "required_tools": []}}
  ]
}}"""

    try:
        result = llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        data = json.loads(_extract_json(text))
        validated = PlannedTaskList(**data)
        logger.info("LLM Planner generated %d tasks, intent=%s", len(validated.planned_tasks), validated.intent)
        return {"intent": validated.intent, "planned_tasks": [t.model_dump() for t in validated.planned_tasks]}
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        logger.warning("LLM Planner parse failed: %s, falling back", e)
        return _fallback_planner(state)
    except Exception:
        logger.exception("LLM Planner failed, falling back")
        return _fallback_planner(state)
