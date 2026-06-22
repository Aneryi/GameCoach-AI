"""工具层 — LangChain @tool 封装的游戏数据查询接口。

所有工具通过 @tool 装饰器注册，支持 LLM tool calling 和 LangSmith 追踪。
每个工具都有 status 字段区分"正常返回"和"不可用"，防止上层误判。
"""

from gamecoach.tools.guide_rag import guide_rag_tool
from gamecoach.tools.hero_database import character_database_tool
from gamecoach.tools.items import build_tool
from gamecoach.tools.match_history import match_history_tool
from gamecoach.tools.patch_meta import patch_meta_tool

__all__ = [
    "match_history_tool",
    "character_database_tool",
    "patch_meta_tool",
    "guide_rag_tool",
    "build_tool",
]
