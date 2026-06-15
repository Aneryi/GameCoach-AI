from __future__ import annotations

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from gamecoach.config.settings import get_settings

logger = logging.getLogger(__name__)

_HF_EMBEDDINGS: Optional[object] = None


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
    """获取本地 HuggingFace embedding 模型。

    使用 all-MiniLM-L6-v2 做本地 embedding，不依赖外部 API。
    模型首次加载时会自动下载（约 80MB），后续调用复用缓存的实例。

    Returns:
        HuggingFaceEmbeddings 实例。
    """
    global _HF_EMBEDDINGS
    if _HF_EMBEDDINGS is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        _HF_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _HF_EMBEDDINGS
