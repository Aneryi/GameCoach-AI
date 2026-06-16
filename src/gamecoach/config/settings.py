from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 自动加载项目根目录的 .env 文件
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
load_dotenv(_env_path, override=False)


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str | None
    deepseek_model: str
    deepseek_base_url: str
    dashscope_api_key: str | None
    langchain_project: str
    langchain_tracing_v2: str


def get_settings() -> Settings:
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "gamecoach-ai"),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "true"),
    )
