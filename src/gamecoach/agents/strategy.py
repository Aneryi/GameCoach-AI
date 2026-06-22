"""
LLM-Powered Strategy Agent。

合成战绩分析、玩家画像、角色推荐和 RAG 检索结果，
生成结构化的策略建议和可执行的训练计划。

输入（从 state 读取）：
- match_analysis: 战绩指标和诊断弱点
- memory: 玩家画像（段位、偏好、历史弱点）
- character_recommendations: 推荐角色列表
- rag_context: RAG 检索到的攻略片段

输出（写入 state）：
- strategy: 策略建议（diagnosis, priorities, action_items, avoid_items）
- training_plan: 训练计划（duration_days, daily_tasks, review_checkpoints）
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
    """训练计划中的一天。"""
    day: int                          # 第几天，从 1 开始
    theme: str                        # 训练主题，如 "减少无效死亡"
    tasks: list[str]                  # 具体训练任务
    success_criteria: list[str]       # 完成标准，可量化验证


class TrainingPlan(BaseModel):
    """训练计划。duration_days 在 3/7/14 中选择。"""
    duration_days: int = Field(description="训练天数，3/7/14")
    goal: str                         # 训练目标
    daily_tasks: list[DailyTask]      # 每日任务
    review_checkpoints: list[str]     # 复盘检查点


class StrategyOutput(BaseModel):
    """
    策略建议的完整输出。

    - diagnosis: 核心问题诊断（2-3句）
    - priorities: 改进优先级列表
    - action_items: 可执行的具体动作（每个要具体到场景）
    - avoid_items: 应避免的行为
    - training_plan: 阶段性训练计划
    """
    diagnosis: str
    priorities: list[str]
    action_items: list[str]
    avoid_items: list[str]
    training_plan: TrainingPlan


# ── 辅助函数 ──

def _has_chinese(text: str) -> bool:
    """
    检测文本是否含中文字符。

    用于 fallback 模式下自动选择输出语言。
    判断逻辑：扫描 Unicode 范围 U+4E00–U+9FFF（CJK 统一表意文字）。
    """
    return any('一' <= c <= '鿿' for c in text)


def _extract_json(text: str) -> str:
    """从 LLM 自由文本输出中提取 JSON 字符串。"""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0).strip()
    return text.strip()


# ── Fallback Strategy ──

def _fallback_strategy(state: GameCoachState) -> GameCoachState:
    """
    LLM 不可用时的规则版策略生成器。

    自动检测用户输入语言（中文/英文），
    返回对应语言的硬编码策略模板和 3 天训练计划。
    """
    analysis = state.get("match_analysis", {})
    memory = state.get("memory", {})
    weaknesses = analysis.get("weaknesses", []) + memory.get("weaknesses", [])
    user_msg = state.get("normalized_message", state.get("user_message", ""))
    cn = _has_chinese(user_msg)

    training_plan = {
        "duration_days": 3,
        "goal": "先把胜率拉回 50% 以上" if cn else "Stabilize win rate above 50%",
        "daily_tasks": [
            {
                "day": 1,
                "theme": "减少无效死亡" if cn else "Reduce unnecessary deaths",
                "tasks": [
                    "每次死亡后记录前 3 次原因" if cn else "Record first 3 death causes each game",
                    "10 分钟后没有视野不单带过河道" if cn else "Avoid pushing without vision after 10 min",
                ],
                "success_criteria": ["单局死亡 <= 5" if cn else "Deaths <= 5 per game"],
            },
            {
                "day": 2,
                "theme": "团战站位" if cn else "Teamfight positioning",
                "tasks": [
                    "保持在前排后方输出" if cn else "Stay behind frontline",
                    "等敌方关键技能交出后再进场" if cn else "Wait for enemy engage before committing",
                ],
                "success_criteria": [
                    "参团率 >= 55%" if cn else "Participation >= 55%",
                    "团战先手死亡次数为 0" if cn else "Zero first-deaths in teamfights",
                ],
            },
            {
                "day": 3,
                "theme": "固定角色池" if cn else "Narrow character pool",
                "tasks": [
                    "只使用 2 个推荐角色排位" if cn else "Only use 2 recommended characters",
                    "复盘胜负局的经济差和死亡点" if cn else "Review economy and death points after each game",
                ],
                "success_criteria": [
                    "连续 3 局使用同一套复盘模板" if cn else "3 consecutive games on same characters",
                ],
            },
        ],
        "review_checkpoints": [
            "第 3 天复盘 KDA" if cn else "Day 3: review KDA",
            "对比本周胜率变化" if cn else "Compare win rate change",
        ],
    }

    strategy = {
        "diagnosis": (
            "当前优先级不是扩角色池，而是降低中期掉点和团战暴毙。" if cn
            else "Priority is reducing mid-game deaths and teamfight mistakes."
        ),
        "priorities": [
            "少死" if cn else "Die less",
            "固定角色池" if cn else "Fix character pool",
            "提高中期参团率" if cn else "Improve teamfight participation",
        ],
        "weaknesses": weaknesses,
        "action_items": [
            "10 分钟后只吃安全线，队友不在附近时不压深线。" if cn
            else "Only farm safe lanes after 10 min.",
            "团战保持在前排后方输出，敌方突进未露头前不交位移向前。" if cn
            else "Stay behind frontline; don't use mobility forward until enemy dive is revealed.",
            "先用 2 个容错较高角色打 20 场，避免频繁换角色导致复盘失效。" if cn
            else "Play 20 games on 2 comfortable characters before switching.",
        ],
    }
    return {"strategy": strategy, "training_plan": training_plan}


# ── LLM Strategy ──

def create_llm_strategy(state: GameCoachState) -> GameCoachState:
    """
    LLM 驱动的策略生成。

    将战绩数据、玩家画像、角色推荐和 RAG 结果注入 prompt，
    让 LLM 综合判断后生成结构化策略建议。

    为什么 training_plan 由 Strategy Agent 生成：
    训练计划是对策略的具体化执行方案——"少死"的策略对应的训练是
    "记录死亡原因"和"不无视野带线"。两者天然绑定，分开用两个 Agent
    会产生不一致的风险（策略说少死，训练计划却主练输出）。
    """
    llm = get_chat_model(temperature=0.3)
    if llm is None:
        logger.info("LLM 不可用，使用 fallback strategy。")
        return _fallback_strategy(state)

    # ── 收集上下文 ──
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

    # ── 角色推荐摘要 ──
    char_text = ""
    if recommendations:
        char_text = "\n推荐角色："
        for r in recommendations[:3]:
            char_text += f"\n- {r.get('character', '?')}: {'; '.join(r.get('fit_reasons', []))}"

    # ── RAG 攻略摘要 ──
    rag_text = ""
    if rag_context:
        snippets = [d.get("snippet", "") for d in rag_context[:3] if d.get("snippet")]
        if snippets:
            rag_text = "\n相关攻略参考：\n" + "\n".join(f"- {s}" for s in snippets)

    # ── LLM 调用 ──
    prompt = f"""你是游戏教练的策略生成器。根据玩家的数据，输出 JSON 格式的策略建议和训练计划。

玩家问题：{user_msg}

数据依据：
- 最近胜率：{win_rate:.0%}
- 平均 KDA：{avg_kda}
- 平均死亡：{avg_deaths}
- 参团率：{participation:.0%}
- 分析弱点：{', '.join(analysis_weaknesses) if analysis_weaknesses else '无'}
- 玩家弱点：{', '.join(memory_weaknesses) if memory_weaknesses else '无'}
- 优势：{', '.join(strengths) if strengths else '无'}
{char_text}
{rag_text}

要求输出 JSON（只输出 JSON，不要其他文字）：
{{
  "diagnosis": "核心问题诊断（2-3句话）",
  "priorities": ["优先级1", "优先级2", "优先级3"],
  "action_items": ["具体可执行动作1", "动作2", "动作3"],
  "avoid_items": ["应避免的行为1", "行为2"],
  "training_plan": {{
    "duration_days": 3,
    "goal": "训练目标",
    "daily_tasks": [
      {{"day": 1, "theme": "训练主题", "tasks": ["任务1", "任务2"], "success_criteria": ["完成标准"]}}
    ],
    "review_checkpoints": ["检查点1", "检查点2"]
  }}
}}

规则：
- training_plan.duration_days：简单问题 3 天，一般 7 天，复杂 14 天
- 每天训练主题不重复，任务具体可验证
- 用和用户输入相同的语言回复"""

    try:
        result = llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        data = json.loads(_extract_json(text))
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
