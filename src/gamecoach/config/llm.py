from __future__ import annotations

from langchain_openai import ChatOpenAI

from gamecoach.config.settings import get_settings


def get_chat_model(temperature: float = 0.2) -> ChatOpenAI:
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Configure it in your system environment "
            "or put it in a local .env file."
        )

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
    )

