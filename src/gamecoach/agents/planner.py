"""LLM-Powered Planner Agent。

使用 DeepSeek LLM 做任务规划，将用户的自然语言问题拆解为 PlannedTask 列表。
LLM 不可用时自动回退到关键词匹配规则。
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field, ValidationError

from gamecoach.config.llm import get_chat_model
from gamecoach.graph.state import GameCoachState

logger = logging.getLogger(__name__)

# ── Pydantic 输出模型 ──


class PlannedTaskModel(BaseModel):
    task_id: str = Field(description="任务编号，如 t1, t2")
    task_type: str = Field(
        description=(
            "任务类型: match_analysis(战绩分析), hero_recommendation(英雄推荐), "
            "build_recommendation(出装推荐), strategy_generation(策略生成), "
            "rag_lookup(攻略检索), memory_lookup(读取玩家画像), "
            "training_plan(训练计划)"
        )
    )
    description: str = Field(description="一句话描述任务做什么")
    priority: int = Field(description="优先级，1最高")
    required_tools: list[str] = Field(
        description="需要的工具: match_history_tool / hero_database_tool / patch_meta_tool / guide_rag_tool"
    )


class PlannedTaskList(BaseModel):
    intent: str = Field(description="用户意图分类")
    planned_tasks: list[PlannedTaskModel] = Field(description="拆解后的任务列表")


# ── Fallback Planner ──


def _fallback_planner(state: GameCoachState) -> GameCoachState:
    """LLM 不可用时的规则版 Planner。"""
    message = state.get("normalized_message", state.get("user_message", ""))
    tasks = [
        PlannedTaskModel(
            task_id="t1",
            task_type="memory_lookup",
            description="读取玩家长期画像，用于个性化建议。",
            priority=1,
            required_tools=["player_memory_store"],
        ),
        PlannedTaskModel(
            task_id="t2",
            task_type="match_analysis",
            description="分析最近 20 场战绩，定位胜率、KDA、死亡和参团问题。",
            priority=2,
            required_tools=["match_history_tool"],
        ),
        PlannedTaskModel(
            task_id="t3",
            task_type="hero_recommendation",
            description="结合英雄池与当前版本推荐上分英雄。",
            priority=3,
            required_tools=["hero_database_tool", "patch_meta_tool"],
        ),
        PlannedTaskModel(
            task_id="t4",
            task_type="strategy_generation",
            description="生成打法建议和短期训练重点。",
            priority=4,
            required_tools=[],
        ),
    ]
    intent = "improve_rank"
    if "英雄" in message or "练什么" in message:
        intent = "hero_recommendation"
    if "出装" in message or "装备" in message:
        intent = "build_recommendation"
    if "攻略" in message or "怎么打" in message:
        intent = "strategy_help"

    return {
        "intent": intent,
        "planned_tasks": [t.model_dump() for t in tasks],
    }


# ── JSON 解析辅助 ──


def _extract_json(text: str) -> str:
    """从 LLM 输出中提取 JSON 字符串。"""
    # 尝试匹配 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # 尝试匹配裸 JSON { ... }
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0).strip()
    return text.strip()


# ── LLM Planner ──


def create_llm_planner(state: GameCoachState) -> GameCoachState:
    """LLM 驱动的任务规划。

    分析用户的自然语言输入，判断意图并生成结构化任务列表。
    DeepSeek 不支持 structured output，改用 JSON prompt + 手动解析。
    """
    message = state.get("normalized_message", state.get("user_message", ""))
    memory = state.get("memory", {})

    llm = get_chat_model(temperature=0.1)
    if llm is None:
        logger.info("LLM 不可用，使用 fallback planner。")
        return _fallback_planner(state)

    memory_context = ""
    if memory:
        rank = memory.get("rank", "未知")
        roles = "、".join(memory.get("main_roles", []))
        goals = "、".join(memory.get("goals", []))
        memory_context = f"\n玩家画像：段位 {rank}，主玩 {roles}，目标 {goals}。"

    prompt = f"""你是游戏成长教练的任务规划器。分析玩家的输入，输出 JSON 格式的任务拆解。

玩家输入：{message}{memory_context}

可用任务类型 task_type：
- match_analysis: 分析战绩数据（胜率、KDA、死亡等）
- hero_recommendation: 推荐适合上分的英雄
- build_recommendation: 推荐装备/出装方案
- strategy_generation: 生成打法和策略建议
- rag_lookup: 检索攻略和版本指南
- memory_lookup: 读取玩家长期画像
- training_plan: 生成阶段性训练计划

规则：
1. 大多数问题需要 memory_lookup 先获取玩家画像
2. 涉及战绩/表现的问题需要 match_analysis
3. 问英雄选择/推荐的需要 hero_recommendation
4. 问装备/出装的需要 build_recommendation
5. 问打法/策略/站位技巧的需要 rag_lookup 和 strategy_generation
6. 问训练计划的需要 training_plan
7. 每个 task_type 最多出现一次
8. 根据用户具体问题，可能不需要所有任务

要求输出如下 JSON 格式（只输出 JSON，不要其他文字）：
{{
  "intent": "improve_rank",
  "planned_tasks": [
    {{
      "task_id": "t1",
      "task_type": "memory_lookup",
      "description": "读取玩家长期画像",
      "priority": 1,
      "required_tools": []
    }}
  ]
}}"""

    try:
        result = llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        json_str = _extract_json(text)
        data = json.loads(json_str)

        # Pydantic 验证
        validated = PlannedTaskList(**data)
        logger.info(
            "LLM Planner 生成 %d 个任务，intent=%s",
            len(validated.planned_tasks),
            validated.intent,
        )
        return {
            "intent": validated.intent,
            "planned_tasks": [t.model_dump() for t in validated.planned_tasks],
        }
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        logger.warning("LLM Planner 输出解析失败: %s，回退到规则版", e)
        return _fallback_planner(state)
    except Exception:
        logger.exception("LLM Planner 调用失败，回退到规则版")
        return _fallback_planner(state)
