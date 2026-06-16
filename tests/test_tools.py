"""Tool layer unit tests."""

from gamecoach.tools.hero_database import character_database_tool
from gamecoach.tools.match_history import match_history_tool
from gamecoach.tools.patch_meta import patch_meta_tool


class TestMatchHistoryTool:
    def test_returns_data_for_player_001(self):
        result = match_history_tool.invoke({"player_id": "player_001", "limit": 5})
        assert result["status"] == "ok"
        assert len(result["matches"]) == 5
        assert "character" in result["matches"][0]

    def test_limit_respected(self):
        result = match_history_tool.invoke({"player_id": "player_001", "limit": 3})
        assert len(result["matches"]) == 3

    def test_default_limit(self):
        result = match_history_tool.invoke({"player_id": "player_001"})
        assert len(result["matches"]) <= 20


class TestCharacterDatabaseTool:
    def test_returns_all_characters_when_no_filter(self):
        result = character_database_tool.invoke({})
        assert result["status"] == "ok"
        assert "characters" in result
        assert "Alpha" in result["characters"]

    def test_returns_specific_character(self):
        result = character_database_tool.invoke({"name": "Alpha"})
        assert result["status"] == "ok"
        assert result["data"]["role"] == "damage"
        assert result["data"]["difficulty"] == "medium"

    def test_unknown_character_returns_unavailable(self):
        result = character_database_tool.invoke({"name": "NonExistent"})
        assert result["status"] == "unavailable"


class TestPatchMetaTool:
    def test_returns_meta_for_role(self):
        result = patch_meta_tool.invoke({"role": "damage"})
        assert result["status"] == "ok"
        assert "strong_characters" in result
        assert "Alpha" in result["strong_characters"]

    def test_returns_meta_with_defaults(self):
        result = patch_meta_tool.invoke({})
        assert result["status"] == "ok"
        assert result["role"] == "damage"
