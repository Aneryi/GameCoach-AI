from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# RAG retriever 的惰性引用，避免循环导入
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        try:
            from gamecoach.rag.retriever import retrieve as _retrieve

            _retriever = _retrieve
        except ImportError:
            logger.warning("RAG 模块未安装，guide_rag_tool 将返回空结果。")
            _retriever = lambda query, top_k=5: []  # noqa: E731
    return _retriever


@tool
def guide_rag_tool(query: str, top_k: int = 5) -> dict[str, Any]:
    """检索游戏攻略、版本指南和职业打法相关内容。

    适用于查询团战站位、英雄连招、出装思路、对线技巧、
    版本强势打法、地图资源控制等攻略性问题。

    Args:
        query: 自然语言搜索查询，例如 "teamfight positioning tips"。
        top_k: 返回结果数量，默认 5。

    Returns:
        status 为 "ok" 时 documents 包含检索到的攻略片段列表，
        每条含 source、title、score、snippet。
    """
    try:
        retrieve_fn = _get_retriever()
        docs = retrieve_fn(query, top_k=top_k)
        return {
            "status": "ok",
            "query": query,
            "top_k": top_k,
            "documents": docs,
        }
    except Exception:
        logger.exception("攻略检索失败")
        return {"status": "unavailable", "documents": [], "reason": "攻略检索暂不可用"}
