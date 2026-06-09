from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str | None
    deepseek_model: str
    deepseek_base_url: str
    langchain_project: str
    langchain_tracing_v2: str


def get_settings() -> Settings:
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "gamecoach-ai"),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "true"),
    )

