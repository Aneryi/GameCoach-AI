from __future__ import annotations

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from gamecoach.config.settings import get_settings

logger = logging.getLogger(__name__)

_DASH_SCOPE_EMBEDDINGS: Optional[object] = None


def get_chat_model(
    temperature: float = 0.2,
) -> Optional[ChatOpenAI]:
    """创建指向 DeepSeek API 的 ChatOpenAI 实例。

    DeepSeek API 兼容 OpenAI 接口格式，可通过 langchain-openai 的 ChatOpenAI 调用。

    Returns:
        ChatOpenAI 实例，如果 DEEPSEEK_API_KEY 未配置则返回 None。
    """
    settings = get_settings()
    if not settings.deepseek_api_key:
        logger.warning("DEEPSEEK_API_KEY 未配置，LLM 调用将使用 fallback 模式。")
        return None

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
    )


def get_embeddings():
    """获取阿里 DashScope embedding 模型。

    使用 text-embedding-v2，API Key 从环境变量 DASHSCOPE_API_KEY 读取。
    实例会被缓存复用，避免重复创建。

    Returns:
        DashScopeEmbeddings 实例。如果 DASHSCOPE_API_KEY 未配置则返回 None。
    """
    global _DASH_SCOPE_EMBEDDINGS
    if _DASH_SCOPE_EMBEDDINGS is None:
        from langchain_community.embeddings import DashScopeEmbeddings

        settings = get_settings()
        if not settings.dashscope_api_key:
            logger.warning("DASHSCOPE_API_KEY 未配置，RAG 检索将返回空结果。")
            return None

        _DASH_SCOPE_EMBEDDINGS = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=settings.dashscope_api_key,
        )
    return _DASH_SCOPE_EMBEDDINGS
