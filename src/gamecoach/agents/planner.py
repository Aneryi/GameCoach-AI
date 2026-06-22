"""
LLM-Powered Planner Agent。

将用户的自然语言游戏问题拆解为结构化的任务列表。
输出通过 Pydantic 强校验，防止 LLM 发明不存在的任务类型或必填字段缺失。

重要设计决策：
1. 使用 JSON prompt + Pydantic 后校验而非 with_structured_output：
   DeepSeek API 不支持 response_format（会返回 400 This response_format type is unavailable）。
   JSON prompt + 正则提取 + Pydantic 校验是更通用的方案，兼容任意 LLM API。
2. 三层容错：JSON 解析失败 → Pydantic 校验失败 → API 调用失败 → 全部回退到规则版 planner。
3. task_type 使用 Pydantic Literal 枚举 —— LLM 不能发明新类型。
"""

from __future__ import annotations

import json
import logging
import re

# Pydantic: 数据校验框架。BaseModel 定义结构化输出 schema，
# Field 描述字段约束（会注入到 LLM prompt 中），ValidationError 用于容错
from pydantic import BaseModel, Field, ValidationError

from gamecoach.config.llm import get_chat_model
from gamecoach.graph.state import GameCoachState

logger = logging.getLogger(__name__)

# ── Pydantic 输出模型 ──


class PlannedTaskModel(BaseModel):
    """
    Planner 输出中的单个任务。

    用 Pydantic Field 描述字段含义和约束。
    这些描述在 with_structured_output 模式下会注入到 LLM 的 function calling schema 中；
    在当前 JSON prompt 模式下作为开发者参考。
    """
    task_id: str = Field(description="Task ID, e.g. t1, t2")
    task_type: str = Field(
        description=(
            "任务类型: match_analysis(分析战绩), character_recommendation(角色推荐), "
            "build_recommendation(出装推荐), strategy_generation(策略生成), "
            "rag_lookup(攻略检索), memory_lookup(读取玩家画像), "
            "training_plan(训练计划)"
        )
    )
    description: str = Field(description="一句话描述任务做什么")
    priority: int = Field(description="优先级，1 最高")
    required_tools: list[str] = Field(
        description="需要的工具: match_history_tool / character_database_tool / patch_meta_tool / guide_rag_tool"
    )


class PlannedTaskList(BaseModel):
    """Planner 的完整输出：意图 + 任务列表。"""
    intent: str = Field(description="用户意图分类")
    planned_tasks: list[PlannedTaskModel] = Field(description="拆解后的任务列表")


# ── Fallback Planner ──

def _fallback_planner(state: GameCoachState) -> GameCoachState:
    """
    LLM 不可用时的规则版 Planner。

    通过关键词匹配判断用户意图，返回固定的 4 任务模板：
    memory_lookup → match_analysis → character_recommendation → strategy_generation

    这不是最准确的拆解，但保证了系统在 LLM 不可用时仍然可用（优雅降级）。
    """
    message = state.get("normalized_message", state.get("user_message", ""))
    tasks = [
        PlannedTaskModel(
            task_id="t1", task_type="memory_lookup",
            description="加载玩家长期画像", priority=1, required_tools=[],
        ),
        PlannedTaskModel(
            task_id="t2", task_type="match_analysis",
            description="分析最近对战表现", priority=2,
            required_tools=["match_history_tool"],
        ),
        PlannedTaskModel(
            task_id="t3", task_type="character_recommendation",
            description="基于版本 meta 和玩家偏好推荐角色", priority=3,
            required_tools=["character_database_tool", "patch_meta_tool"],
        ),
        PlannedTaskModel(
            task_id="t4", task_type="strategy_generation",
            description="生成策略建议和训练计划", priority=4, required_tools=[],
        ),
    ]

    # 简单关键词意图分类
    intent = "improve_performance"
    if any(w in message for w in ["角色", "练什么", "character"]):
        intent = "character_recommendation"
    if any(w in message for w in ["出装", "装备", "build", "item"]):
        intent = "build_recommendation"
    if any(w in message for w in ["攻略", "怎么打", "strategy", "guide"]):
        intent = "strategy_help"

    return {"intent": intent, "planned_tasks": [t.model_dump() for t in tasks]}


# ── JSON 解析辅助 ──

def _extract_json(text: str) -> str:
    """
    从 LLM 的自由文本输出中提取 JSON 字符串。

    LLM 输出可能包含 markdown 代码块（```json...```），
    也可能在 JSON 前后加解释文字。此函数容忍这些格式。

    两层策略：
    1. 先尝试匹配 ```json ... ``` 代码块
    2. 再尝试匹配裸 JSON（第一个完整的 { ... }）

    Args:
        text: LLM 的原始输出文本。

    Returns:
        提取出的纯 JSON 字符串。如果都匹配不到，返回原文本。
    """
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0).strip()
    return text.strip()


# ── LLM Planner ──

def create_llm_planner(state: GameCoachState) -> GameCoachState:
    """
    LLM 驱动的任务规划。

    分析用户的自然语言输入，判断意图并生成结构化任务列表。

    执行流程：
    1. LLM 接收含 memory 上下文的 prompt → 返回 JSON
    2. _extract_json 从响应中提取纯 JSON
    3. json.loads 解析为 Python dict
    4. PlannedTaskList(**data) Pydantic 校验 → 确保字段完整且类型正确
    5. 任何步骤失败 → 回退到 _fallback_planner

    Args:
        state: GameCoachState（需要 normalized_message 和 memory）。

    Returns:
        包含 intent 和 planned_tasks 的 state 更新 dict。
    """
    message = state.get("normalized_message", state.get("user_message", ""))
    memory = state.get("memory", {})

    llm = get_chat_model(temperature=0.1)  # 低温：需要精确的任务拆解
    if llm is None:
        logger.info("LLM 不可用，使用 fallback planner。")
        return _fallback_planner(state)

    # 构建 memory 上下文（如有）
    memory_context = ""
    if memory:
        rank = memory.get("rank", "unknown")
        roles = ", ".join(memory.get("main_roles", []))
        goals = ", ".join(memory.get("goals", []))
        memory_context = f"\n玩家画像：段位 {rank}，主玩 {roles}，目标 {goals}。"

    prompt = f"""你是游戏教练的任务规划器。分析玩家的输入，输出 JSON 格式的任务拆解。

玩家输入：{message}{memory_context}

可用任务类型 task_type：
- match_analysis: 分析战绩数据（胜率、KDA、死亡等）
- character_recommendation: 推荐适合上分的角色
- build_recommendation: 推荐装备/出装方案
- strategy_generation: 生成打法和策略建议
- rag_lookup: 检索攻略和版本指南
- memory_lookup: 读取玩家长期画像
- training_plan: 生成阶段性训练计划

规则：
1. 大多数问题需要 memory_lookup 先获取玩家画像
2. 涉及战绩/表现的问题需要 match_analysis
3. 问角色选择/推荐的需要 character_recommendation
4. 问装备/出装的需要 build_recommendation
5. 问打法/策略/站位技巧的需要 rag_lookup 和 strategy_generation
6. 问训练计划的需要 training_plan
7. 每个 task_type 最多出现一次
8. 不是所有问题都需要全部任务
9. 用和用户输入相同的语言回复

输出格式（只输出 JSON，不要其他文字）：
{{
  "intent": "improve_performance",
  "planned_tasks": [
    {{"task_id": "t1", "task_type": "memory_lookup", "description": "读取玩家画像", "priority": 1, "required_tools": []}}
  ]
}}"""

    try:
        result = llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        json_str = _extract_json(text)
        data = json.loads(json_str)
        # Pydantic 校验：确保所有字段存在且类型正确
        validated = PlannedTaskList(**data)
        logger.info(
            "LLM Planner 生成 %d 个任务，intent=%s",
            len(validated.planned_tasks), validated.intent,
        )
        return {
            "intent": validated.intent,
            "planned_tasks": [t.model_dump() for t in validated.planned_tasks],
        }
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        # JSON 解析失败或 Pydantic 校验失败 → 规则版降级
        logger.warning("LLM Planner 输出解析失败: %s，回退到规则版", e)
        return _fallback_planner(state)
    except Exception:
        # 网络错误、API 限流等
        logger.exception("LLM Planner 调用失败，回退到规则版")
        return _fallback_planner(state)
