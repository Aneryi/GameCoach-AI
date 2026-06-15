"""LLM-Powered Strategy Agent。

根据战绩分析、玩家画像、英雄推荐、RAG 检索结果，
生成结构化策略建议和训练计划。
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


class DailyTask(BaseModel):
    day: int = Field(description="第几天")
    theme: str = Field(description="当日训练主题")
    tasks: list[str] = Field(description="具体训练任务列表")
    success_criteria: list[str] = Field(description="完成标准")


class TrainingPlan(BaseModel):
    duration_days: int = Field(description="训练计划总天数，3/7/14")
    goal: str = Field(description="训练目标描述")
    daily_tasks: list[DailyTask] = Field(description="每日任务列表")
    review_checkpoints: list[str] = Field(description="复盘检查点")


class StrategyOutput(BaseModel):
    diagnosis: str = Field(description="核心问题诊断")
    priorities: list[str] = Field(description="改进优先级列表")
    action_items: list[str] = Field(description="可执行的具体动作")
    avoid_items: list[str] = Field(description="应避免的行为")
    training_plan: TrainingPlan = Field(description="训练计划")


# ── JSON 解析辅助 ──


def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0).strip()
    return text.strip()


# ── Fallback Strategy ──


def _fallback_strategy(state: GameCoachState) -> GameCoachState:
    """LLM 不可用时的规则版策略生成器。"""
    analysis = state.get("match_analysis", {})
    memory = state.get("memory", {})
    weaknesses = analysis.get("weaknesses", []) + memory.get("weaknesses", [])

    training_plan = {
        "duration_days": 3,
        "goal": "先把射手位胜率拉回 50% 以上",
        "daily_tasks": [
            {
                "day": 1,
                "theme": "减少无效死亡",
                "tasks": ["10 分钟后没有视野不单带过河道", "每局记录前 3 次死亡原因"],
                "success_criteria": ["单局死亡 <= 5"],
            },
            {
                "day": 2,
                "theme": "团战站位",
                "tasks": ["团战前确认敌方刺客位置", "等关键控制交出后再向前输出"],
                "success_criteria": ["参团率 >= 55%", "团战先手死亡次数为 0"],
            },
            {
                "day": 3,
                "theme": "固定英雄池",
                "tasks": ["只使用 2 个推荐英雄排位", "复盘胜负局的经济差和死亡点"],
                "success_criteria": ["连续 3 局使用同一套复盘模板"],
            },
        ],
        "review_checkpoints": ["第 3 天复盘 KDA", "对比本周胜率变化"],
    }
    strategy = {
        "diagnosis": "当前优先级不是扩英雄池，而是降低中期掉点和团战暴毙。",
        "priorities": ["少死", "固定英雄池", "提高中期参团率"],
        "weaknesses": weaknesses,
        "action_items": [
            "10 分钟后只吃安全线，队友不在附近时不压深线。",
            "团战保持在前排后方输出，敌方突进未露头前不交位移向前。",
            "先用 2 个容错较高英雄打 20 场，避免频繁换英雄导致复盘失效。",
        ],
    }
    return {"strategy": strategy, "training_plan": training_plan}


# ── LLM Strategy ──


def create_llm_strategy(state: GameCoachState) -> GameCoachState:
    """LLM 驱动的策略生成。

    合成战绩分析、玩家画像、英雄推荐和 RAG 结果，
    生成个性化的策略建议和训练计划。
    """
    llm = get_chat_model(temperature=0.3)
    if llm is None:
        logger.info("LLM 不可用，使用 fallback strategy。")
        return _fallback_strategy(state)

    # 收集上下文
    analysis = state.get("match_analysis", {})
    memory = state.get("memory", {})
    recommendations = state.get("hero_recommendations", [])
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

    hero_text = ""
    if recommendations:
        hero_text = "\n推荐英雄："
        for r in recommendations[:3]:
            hero_text += f"\n- {r.get('hero', '未知')}：{'; '.join(r.get('fit_reasons', []))}"

    rag_text = ""
    if rag_context:
        snippets = [d.get("snippet", "") for d in rag_context[:3] if d.get("snippet")]
        if snippets:
            rag_text = "\n相关攻略参考：\n" + "\n".join(f"- {s}" for s in snippets)

    prompt = f"""你是游戏成长教练的策略生成器。根据玩家的数据，输出 JSON 格式的策略建议和训练计划。

玩家问题：{user_msg}

数据依据：
- 最近胜率：{win_rate:.0%}
- 平均 KDA：{avg_kda}
- 平均死亡：{avg_deaths}
- 参团率：{participation:.0%}
- 分析弱点：{', '.join(analysis_weaknesses) if analysis_weaknesses else '无'}
- 玩家弱点：{', '.join(memory_weaknesses) if memory_weaknesses else '无'}
- 优势：{', '.join(strengths) if strengths else '无'}
{hero_text}
{rag_text}

要求输出如下 JSON 格式（只输出 JSON，不要其他文字）：
{{
  "diagnosis": "核心问题诊断（2-3句话，中文）",
  "priorities": ["优先级1", "优先级2", "优先级3"],
  "action_items": ["具体可执行动作1", "动作2", "动作3"],
  "avoid_items": ["应避免的行为1", "行为2"],
  "training_plan": {{
    "duration_days": 3,
    "goal": "训练目标描述",
    "daily_tasks": [
      {{
        "day": 1,
        "theme": "训练主题",
        "tasks": ["任务1", "任务2"],
        "success_criteria": ["完成标准"]
      }}
    ],
    "review_checkpoints": ["检查点1", "检查点2"]
  }}
}}

规则：
1. training_plan.duration_days：简单问题 3 天，一般问题 7 天，复杂问题 14 天
2. 每天训练主题不重复，任务具体可验证
3. priorities 和 action_items 要针对玩家的具体弱点
4. 用中文输出"""

    try:
        result = llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        json_str = _extract_json(text)
        data = json.loads(json_str)
        validated = StrategyOutput(**data)
        logger.info(
            "LLM Strategy 生成 %d 天训练计划", validated.training_plan.duration_days
        )
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
        logger.warning("LLM Strategy 输出解析失败: %s，回退到规则版", e)
        return _fallback_strategy(state)
    except Exception:
        logger.exception("LLM Strategy 调用失败，回退到规则版")
        return _fallback_strategy(state)
