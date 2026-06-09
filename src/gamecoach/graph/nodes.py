from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from gamecoach.graph.state import GameCoachState, PlannedTask
from gamecoach.memory.store import load_player_memory
from gamecoach.tools.hero_database import get_heroes
from gamecoach.tools.match_history import get_match_history
from gamecoach.tools.patch_meta import get_patch_meta


def input_normalizer(state: GameCoachState) -> GameCoachState:
    message = state.get("user_message", "").strip()
    return {
        "normalized_message": message,
        "player_id": state.get("player_id") or "player_001",
        "game": state.get("game") or "moba",
        "errors": state.get("errors", []),
    }


def planner(state: GameCoachState) -> GameCoachState:
    message = state.get("normalized_message", "")
    tasks: list[PlannedTask] = [
        {
            "task_id": "t1",
            "task_type": "memory_lookup",
            "description": "读取玩家长期画像，用于个性化建议。",
            "priority": 1,
            "required_tools": ["player_memory_store"],
        },
        {
            "task_id": "t2",
            "task_type": "match_analysis",
            "description": "分析最近 20 场战绩，定位胜率、KDA、死亡和参团问题。",
            "priority": 2,
            "required_tools": ["match_history_tool"],
        },
        {
            "task_id": "t3",
            "task_type": "hero_recommendation",
            "description": "结合英雄池与当前版本推荐上分英雄。",
            "priority": 3,
            "required_tools": ["hero_database_tool", "patch_meta_tool"],
        },
        {
            "task_id": "t4",
            "task_type": "strategy_generation",
            "description": "生成打法建议和短期训练重点。",
            "priority": 4,
            "required_tools": [],
        },
    ]
    intent = "improve_rank"
    if "英雄" in message or "练什么" in message:
        intent = "hero_recommendation"
    if "出装" in message or "装备" in message:
        intent = "build_recommendation"
    return {"intent": intent, "planned_tasks": tasks}


def memory_loader(state: GameCoachState) -> GameCoachState:
    memory = load_player_memory(state.get("player_id", "player_001"))
    return {"memory": memory}


def match_analysis_agent(state: GameCoachState) -> GameCoachState:
    match_data = get_match_history(
        player_id=state.get("player_id", "player_001"),
        limit=20,
        game=state.get("game", "moba"),
    )
    matches = match_data["matches"]
    wins = [m for m in matches if m["result"] == "win"]
    deaths = [m["deaths"] for m in matches]
    kdas = [(m["kills"] + m["assists"]) / max(m["deaths"], 1) for m in matches]
    participations = [m["teamfight_participation"] for m in matches]

    hero_games: dict[str, list[dict]] = defaultdict(list)
    for match in matches:
        hero_games[match["hero"]].append(match)

    hero_win_rates = {
        hero: round(
            len([m for m in hero_matches if m["result"] == "win"]) / len(hero_matches),
            2,
        )
        for hero, hero_matches in hero_games.items()
    }
    most_played_heroes = [hero for hero, _ in Counter(m["hero"] for m in matches).most_common(3)]

    avg_deaths = mean(deaths)
    avg_participation = mean(participations)
    weaknesses = []
    if avg_deaths >= 6:
        weaknesses.append("平均死亡偏高，团战和中期转线需要降低风险。")
    if avg_participation < 0.55:
        weaknesses.append("参团率偏低，中期容易和队友节奏脱节。")
    if len(wins) / len(matches) < 0.5:
        weaknesses.append("近期胜率低于 50%，需要先固定英雄池并减少高风险决策。")

    analysis = {
        "summary": "最近 20 场表现显示，主要问题集中在死亡偏高和中期参团稳定性不足。",
        "metrics": {
            "matches": len(matches),
            "win_rate": round(len(wins) / len(matches), 2),
            "avg_kda": round(mean(kdas), 2),
            "avg_deaths": round(avg_deaths, 2),
            "teamfight_participation": round(avg_participation, 2),
        },
        "hero_win_rates": hero_win_rates,
        "most_played_heroes": most_played_heroes,
        "weaknesses": weaknesses,
        "strengths": ["射手英雄使用频率高，适合围绕稳定发育和团战输出做专项提升。"],
    }
    return {"match_data": match_data, "match_analysis": analysis}


def hero_recommendation_agent(state: GameCoachState) -> GameCoachState:
    memory = state.get("memory", {})
    patch_meta = get_patch_meta(game=state.get("game", "moba"), role="射手")
    heroes = get_heroes()
    favorite_heroes = set(memory.get("favorite_heroes", []))

    recommendations = []
    for hero_name in patch_meta["strong_heroes"]:
        hero = heroes[hero_name]
        fit_reasons = ["当前版本强势"]
        if hero_name in favorite_heroes:
            fit_reasons.append("符合玩家历史英雄偏好")
        if hero["difficulty"] in {"low", "medium"}:
            fit_reasons.append("操作门槛适中，适合先稳定胜率")
        recommendations.append(
            {
                "hero": hero_name,
                "role": hero["role"],
                "difficulty": hero["difficulty"],
                "fit_reasons": fit_reasons,
                "risks": hero["weaknesses"],
            }
        )

    return {"hero_recommendations": recommendations[:3]}


def strategy_agent(state: GameCoachState) -> GameCoachState:
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


def response_agent(state: GameCoachState) -> GameCoachState:
    metrics = state["match_analysis"]["metrics"]
    recommendations = state.get("hero_recommendations", [])
    strategy = state.get("strategy", {})
    plan = state.get("training_plan", {})

    hero_lines = "\n".join(
        f"- {item['hero']}：{'; '.join(item['fit_reasons'])}" for item in recommendations
    )
    task_lines = "\n".join(
        f"Day {item['day']}：{item['theme']}，{item['tasks'][0]}"
        for item in plan.get("daily_tasks", [])
    )
    final_response = f"""结论：
你的主要上分阻碍是中期死亡偏高和团战站位风险，不是单纯的英雄不够多。

数据依据：
- 最近 20 场胜率：{metrics['win_rate'] * 100:.0f}%
- 平均 KDA：{metrics['avg_kda']}
- 平均死亡：{metrics['avg_deaths']}
- 平均参团率：{metrics['teamfight_participation'] * 100:.0f}%

优先改进：
1. {strategy['action_items'][0]}
2. {strategy['action_items'][1]}
3. {strategy['action_items'][2]}

推荐英雄：
{hero_lines}

3 天训练计划：
{task_lines}
"""
    return {"final_response": final_response}


def evaluation_logger(state: GameCoachState) -> GameCoachState:
    metrics = {
        "planner_task_count": len(state.get("planned_tasks", [])),
        "tool_call_success_rate": 1.0,
        "has_match_analysis": bool(state.get("match_analysis")),
        "has_hero_recommendations": bool(state.get("hero_recommendations")),
        "has_training_plan": bool(state.get("training_plan")),
    }
    return {"metrics": metrics}

