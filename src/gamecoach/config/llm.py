"""
LLM 和 Embedding 模型工厂。

提供全局单例的 ChatOpenAI 和 DashScopeEmbeddings 实例。
所有 LLM 调用的入口，统一管理 API Key、模型选择和容错逻辑。

为什么用工厂函数 + 全局缓存而非每次创建新实例：
1. 避免重复建立 HTTP 连接
2. Embedding 模型实例创建有网络开销（API 验证）
3. 全局缓存确保所有 Agent 节点使用相同的模型配置
"""

from __future__ import annotations

import logging
from typing import Optional

# ChatOpenAI: LangChain 的 OpenAI 兼容聊天模型封装
# DeepSeek API 兼容 OpenAI 接口格式，可以直接用 ChatOpenAI 调用
from langchain_openai import ChatOpenAI

from gamecoach.config.settings import get_settings

logger = logging.getLogger(__name__)

# 全局 Embedding 实例缓存（懒加载）
# Optional[object] 而非具体类型，避免导入时触发 langchain_community 的延迟加载
_DASH_SCOPE_EMBEDDINGS: Optional[object] = None


def get_chat_model(
    temperature: float = 0.2,
) -> Optional[ChatOpenAI]:
    """
    获取 LLM 聊天模型实例。

    底层使用 DeepSeek API（通过 ChatOpenAI 的 base_url 指向 DeepSeek 端点）。
    如果 DEEPSEEK_API_KEY 未配置，返回 None —— 调用方必须处理 None 情况（走 fallback）。

    Args:
        temperature: 生成温度。0.1 用于 Planner（需要精确输出），
                     0.3-0.4 用于 Strategy/Response（需要一定创造性）。

    Returns:
        ChatOpenAI 实例，或 None（API Key 未配置时）。
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
    """
    获取文本 Embedding 模型实例。

    使用阿里云 DashScope text-embedding-v2 API。
    为什么用 DashScope 而非 HuggingFace 本地模型：
    - 本地模型（all-MiniLM-L6-v2）需要从 huggingface.co 下载（~80MB）
    - 境内服务器访问 huggingface.co 经常被墙（getaddrinfo failed）
    - DashScope 是国内云服务，网络可达、不需要科学上网

    实例被全局缓存复用，避免重复创建连接。

    Returns:
        DashScopeEmbeddings 实例，或 None（DASHSCOPE_API_KEY 未配置时）。
    """
    global _DASH_SCOPE_EMBEDDINGS
    if _DASH_SCOPE_EMBEDDINGS is None:
        # 延迟导入：只在第一次调用时加载 langchain_community
        # 避免模块导入时的循环依赖和启动延迟
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
