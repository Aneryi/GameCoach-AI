"""GameCoach AI 的 LangGraph 节点实现。"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from statistics import mean

from gamecoach.agents.planner import create_llm_planner
from gamecoach.agents.response import create_llm_response
from gamecoach.agents.strategy import create_llm_strategy
from gamecoach.config.llm import get_chat_model
from gamecoach.graph.router import EXECUTION_ORDER, FIXED_NODES, TASK_NODE_MAP
from gamecoach.graph.state import GameCoachState
from gamecoach.memory.store import load_player_memory, update_player_memory
from gamecoach.tools.hero_database import get_characters
from gamecoach.tools.match_history import get_match_history
from gamecoach.tools.patch_meta import get_patch_meta

logger = logging.getLogger(__name__)


def input_normalizer(state: GameCoachState) -> GameCoachState:
    message = state.get("user_message", "").strip()
    return {
        "normalized_message": message,
        "player_id": state.get("player_id") or "player_001",
        "errors": state.get("errors", []),
        "degraded_nodes": [],
    }


def planner(state: GameCoachState) -> GameCoachState:
    result = create_llm_planner(state)
    planned = result.get("planned_tasks", [])
    required_nodes: set[str] = set()
    for task in planned:
        node = TASK_NODE_MAP.get(task["task_type"])
        if node:
            required_nodes.add(node)

    decisions = {}
    for node in EXECUTION_ORDER:
        decisions[node] = "execute" if node in required_nodes else "skip"
    for node in FIXED_NODES:
        decisions[node] = "fixed"

    result["routing_decisions"] = decisions
    return result


def memory_loader(state: GameCoachState) -> GameCoachState:
    player_id = state.get("player_id", "player_001")
    try:
        memory = load_player_memory(player_id)
    except Exception:
        logger.exception("Memory load failed")
        memory = {}
        state.setdefault("errors", []).append("Player memory load failed")
    return {"memory": memory}


def match_analysis_agent(state: GameCoachState) -> GameCoachState:
    player_id = state.get("player_id", "player_001")
    try:
        match_data = get_match_history(player_id=player_id, limit=20)
    except Exception:
        logger.exception("Match query failed")
        return {
            "match_data": {"matches": []},
            "match_analysis": {"summary": "Data unavailable", "metrics": {}, "weaknesses": [], "strengths": []},
        }

    matches = match_data["matches"]
    if not matches:
        return {
            "match_data": match_data,
            "match_analysis": {"summary": "No match data", "metrics": {"matches": 0}, "weaknesses": [], "strengths": []},
        }

    wins = [m for m in matches if m["result"] == "win"]
    deaths = [m["deaths"] for m in matches]
    kdas = [(m["kills"] + m["assists"]) / max(m["deaths"], 1) for m in matches]
    participations = [m["teamfight_participation"] for m in matches]

    char_games: dict[str, list[dict]] = defaultdict(list)
    for match in matches:
        char_games[match["character"]].append(match)

    char_win_rates = {
        char: round(len([m for m in cms if m["result"] == "win"]) / len(cms), 2)
        for char, cms in char_games.items()
    }
    most_played = [char for char, _ in Counter(m["character"] for m in matches).most_common(3)]

    avg_deaths = mean(deaths)
    avg_participation = mean(participations)
    win_rate = len(wins) / len(matches)

    weaknesses = []
    if avg_deaths >= 6:
        weaknesses.append("Average deaths too high – reduce risky positioning.")
    if avg_participation < 0.55:
        weaknesses.append("Teamfight participation low – improve mid-game rotation timing.")
    if win_rate < 0.5:
        weaknesses.append("Win rate below 50% – narrow character pool and reduce high-risk plays.")

    analysis = {
        "summary": f"Analysis of {len(matches)} recent matches complete.",
        "metrics": {
            "matches": len(matches),
            "win_rate": round(win_rate, 2),
            "avg_kda": round(mean(kdas), 2),
            "avg_deaths": round(avg_deaths, 2),
            "teamfight_participation": round(avg_participation, 2),
        },
        "character_win_rates": char_win_rates,
        "most_played_characters": most_played,
        "weaknesses": weaknesses,
        "strengths": ["Consistent main-role usage provides a solid foundation for targeted improvement."],
    }
    return {"match_data": match_data, "match_analysis": analysis}


def character_recommendation_agent(state: GameCoachState) -> GameCoachState:
    memory = state.get("memory", {})
    main_roles = memory.get("main_roles", ["damage"])
    primary_role = main_roles[0] if main_roles else "damage"

    try:
        patch_meta = get_patch_meta(role=primary_role)
    except Exception:
        logger.exception("Patch data query failed")
        return {"character_recommendations": []}

    try:
        characters = get_characters()
    except Exception:
        logger.exception("Character database query failed")
        characters = {}

    favorite_characters = set(memory.get("favorite_characters", []))

    recommendations = []
    for char_name in patch_meta.get("strong_characters", []):
        char = characters.get(char_name)
        if not char:
            continue
        fit_reasons = ["Strong in current meta"]
        if char_name in favorite_characters:
            fit_reasons.append("Matches player character preference")
        if char.get("difficulty") in {"low", "medium"}:
            fit_reasons.append("Manageable difficulty for consistent results")
        recommendations.append({
            "character": char_name,
            "role": char.get("role", "?"),
            "difficulty": char.get("difficulty", "medium"),
            "fit_reasons": fit_reasons,
            "risks": char.get("weaknesses", []),
        })

    return {"character_recommendations": recommendations[:3]}


def build_agent(state: GameCoachState) -> GameCoachState:
    recommendations = state.get("character_recommendations", [])
    memory = state.get("memory", {})
    main_roles = memory.get("main_roles", ["damage"])
    role = main_roles[0] if main_roles else "damage"

    try:
        from gamecoach.tools.items import get_all_items
        all_items = get_all_items()
    except Exception:
        logger.exception("Item data query failed")
        return {"build_recommendations": []}

    llm = get_chat_model(temperature=0.2)
    char_names = [r.get("character", "") for r in recommendations[:2]] if recommendations else ["any"]

    # Fallback: rule-based recommendation
    if llm is None:
        boots = [{"name": k, **v} for k, v in all_items.items() if v["category"] == "boots"][:1]
        attack = [{"name": k, **v} for k, v in all_items.items() if v["category"] == "attack"][:4]
        defense = [{"name": k, **v} for k, v in all_items.items() if v["category"] == "defense"][:1]
        fallback = [i["name"] for i in boots + attack + defense]
        return {"build_recommendations": [{"character": char_names[0], "scenario": "balanced", "items": fallback, "rationale": "通用角色出装推荐。"}]}

    items_info = "\n".join(
        f"- {name} ({info.get('category', '')}): {info.get('stat', '')}, {info.get('note', '')}"
        for name, info in all_items.items()
    )
    prompt = f"""You are a build advisor. Recommend an equipment set for {', '.join(char_names)} (role: {role}).

Available items:
{items_info}

Reply in this format:
Build: Item1 -> Item2 -> Item3 -> Item4 -> Item5 -> Item6
Rationale: (2-3 sentences why)
Scenario: (balanced / ahead / behind)"""

    try:
        result = llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        import re
        items_match = re.search(r"Build:\s*(.+)", text)
        items_str = items_match.group(1) if items_match else ""
        items_list = [i.strip() for i in items_str.split("->") if i.strip()]
        rationale_match = re.search(r"Rationale:\s*(.+)", text)
        rationale = rationale_match.group(1) if rationale_match else text[:100]
        scenario_match = re.search(r"Scenario:\s*(.+)", text)
        scenario = scenario_match.group(1).strip() if scenario_match else "balanced"
        return {"build_recommendations": [{"character": char_names[0], "scenario": scenario, "items": items_list, "rationale": rationale}]}
    except Exception:
        logger.exception("LLM build recommendation failed")
        return {"build_recommendations": []}


def rag_agent(state: GameCoachState) -> GameCoachState:
    user_msg = state.get("normalized_message", state.get("user_message", ""))
    analysis = state.get("match_analysis", {})
    query_parts = [user_msg]
    for w in analysis.get("weaknesses", []):
        query_parts.append(w)
    query = " ".join(query_parts)
    try:
        from gamecoach.rag.retriever import retrieve
        docs = retrieve(query, top_k=5)
    except ImportError:
        logger.warning("RAG module not available")
        docs = []
    except Exception:
        logger.exception("RAG retrieval failed")
        docs = []
    return {"rag_context": docs}


def strategy_agent(state: GameCoachState) -> GameCoachState:
    result = create_llm_strategy(state)
    strategy = result.get("strategy", {})
    if strategy.get("weaknesses"):
        try:
            update_player_memory(state.get("player_id", "player_001"), {
                "weaknesses": strategy["weaknesses"],
                "source": "match_analysis",
            })
        except Exception:
            logger.debug("Memory update skipped (non-fatal)")
    return result


def response_agent(state: GameCoachState) -> GameCoachState:
    return create_llm_response(state)


def evaluation_logger(state: GameCoachState) -> GameCoachState:
    routing = state.get("routing_decisions", {})
    executed = [k for k, v in routing.items() if v in ("execute", "fixed")]
    metrics = {
        "planner_task_count": len(state.get("planned_tasks", [])),
        "executed_nodes": executed,
        "routing_path": " -> ".join(executed),
        "degraded_node_count": len(state.get("degraded_nodes", [])),
        "tool_call_success_rate": 1.0,
        "has_match_analysis": bool(state.get("match_analysis", {}).get("metrics")),
        "has_character_recommendations": bool(state.get("character_recommendations")),
        "has_build_recommendations": bool(state.get("build_recommendations")),
        "has_training_plan": bool(state.get("training_plan", {}).get("daily_tasks")),
        "rag_hit_count": len(state.get("rag_context", [])),
        "response_length": len(state.get("final_response", "")),
    }
    return {"metrics": metrics}
