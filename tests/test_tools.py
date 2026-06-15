"""工具层单元测试。"""

from gamecoach.tools.hero_database import hero_database_tool
from gamecoach.tools.match_history import match_history_tool
from gamecoach.tools.patch_meta import patch_meta_tool


class TestMatchHistoryTool:
    def test_returns_data_for_player_001(self):
        result = match_history_tool.invoke({"player_id": "player_001", "limit": 5})
        assert result["status"] == "ok"
        assert len(result["matches"]) == 5
        assert "hero" in result["matches"][0]

    def test_limit_respected(self):
        result = match_history_tool.invoke({"player_id": "player_001", "limit": 3})
        assert len(result["matches"]) == 3

    def test_default_limit(self):
        result = match_history_tool.invoke({"player_id": "player_001"})
        assert len(result["matches"]) <= 20


class TestHeroDatabaseTool:
    def test_returns_all_heroes_when_no_filter(self):
        result = hero_database_tool.invoke({})
        assert result["status"] == "ok"
        assert "heroes" in result
        assert "狄仁杰" in result["heroes"]

    def test_returns_specific_hero(self):
        result = hero_database_tool.invoke({"hero": "狄仁杰"})
        assert result["status"] == "ok"
        assert result["data"]["role"] == "射手"
        assert result["data"]["difficulty"] == "medium"

    def test_unknown_hero_returns_unavailable(self):
        result = hero_database_tool.invoke({"hero": "不存在的英雄"})
        assert result["status"] == "unavailable"


class TestPatchMetaTool:
    def test_returns_meta_for_role(self):
        result = patch_meta_tool.invoke({"game": "moba", "role": "射手"})
        assert result["status"] == "ok"
        assert "strong_heroes" in result
        assert "狄仁杰" in result["strong_heroes"]

    def test_returns_meta_with_defaults(self):
        result = patch_meta_tool.invoke({})
        assert result["status"] == "ok"
        assert result["role"] == "射手"
