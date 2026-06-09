from gamecoach.config.settings import get_settings


def test_deepseek_settings_read_system_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "gamecoach-ai-test")

    settings = get_settings()

    assert settings.deepseek_api_key == "test-key"
    assert settings.deepseek_model == "deepseek-chat"
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.langchain_project == "gamecoach-ai-test"

