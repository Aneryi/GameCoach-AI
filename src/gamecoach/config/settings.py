"""
项目配置管理。

负责加载环境变量（.env 文件 + 系统环境变量），
并通过不可变的 Settings 数据类暴露给其他模块。

使用 python-dotenv 在模块加载时自动读取 .env 文件，
不再需要用户手动调用 load_dotenv()。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# python-dotenv: 将 .env 文件中的键值对加载到 os.environ
# override=False 表示系统环境变量优先级高于 .env 文件
from dotenv import load_dotenv

# 定位项目根目录的 .env 文件
# __file__ → settings.py 所在目录 → 上溯三级到项目根
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
load_dotenv(_env_path, override=False)


@dataclass(frozen=True)  # frozen=True: 不可变，防止运行时意外修改配置
class Settings:
    """
    应用全局配置。

    所有字段通过 os.getenv 从环境变量读取，提供默认值。
    frozen=True 确保配置一旦创建就不能被修改（线程安全）。
    """

    # DeepSeek LLM 配置（兼容 OpenAI 接口格式）
    deepseek_api_key: str | None  # None 时 LLM 功能降级为 fallback 模式
    deepseek_model: str            # 默认 deepseek-chat
    deepseek_base_url: str         # 默认 https://api.deepseek.com

    # 阿里云 DashScope Embedding 配置（用于 RAG 向量检索）
    dashscope_api_key: str | None  # None 时 RAG 功能返回空结果

    # LangSmith 追踪配置
    langchain_project: str         # 默认 gamecoach-ai
    langchain_tracing_v2: str      # 默认 "true"，启用 LangSmith Tracing


def get_settings() -> Settings:
    """
    获取全局配置单例。

    每次调用都从环境变量读取最新值（数据类本身不可变，但值来源是动态的）。
    如果需要在运行时修改配置，修改环境变量后重新调用即可。
    """
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "gamecoach-ai"),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "true"),
    )
