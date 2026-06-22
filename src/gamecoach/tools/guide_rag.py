"""
攻略检索工具。

通过 RAG 模块（FAISS + DashScope Embedding）检索攻略文档中的相关内容。
RAG 模块未安装或 Embedding 服务不可用时返回空结果（status="unavailable"）。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# RAG retriever 的惰性引用 —— 避免模块导入时的循环依赖
# 只有在 guide_rag_tool 首次被调用时才加载 retriever
_retriever = None


def _get_retriever():
    """惰性获取 RAG retriever 函数。"""
    global _retriever
    if _retriever is None:
        try:
            from gamecoach.rag.retriever import retrieve as _retrieve
            _retriever = _retrieve
        except ImportError:
            logger.warning("RAG 模块未安装，guide_rag_tool 将返回空结果。")
            _retriever = lambda query, top_k=5: []
    return _retriever


@tool
def guide_rag_tool(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    检索游戏攻略、版本指南和打法教学内容。

    适用于查询站位技巧、角色连招、出装思路、对线技巧、版本强势打法等。
    检索结果基于 FAISS 向量相似度，返回最相关的攻略片段。

    Args:
        query: 自然语言搜索查询，例如 "团队站位技巧"。
        top_k: 返回结果数量，默认 5。

    Returns:
        status="ok" 时 documents 包含检索到的攻略片段（source, title, score, snippet）。
        status="unavailable" 时表示 RAG 服务不可用。
    """
    try:
        retrieve_fn = _get_retriever()
        docs = retrieve_fn(query, top_k=top_k)
        return {"status": "ok", "query": query, "top_k": top_k, "documents": docs}
    except Exception:
        logger.exception("攻略检索失败")
        return {"status": "unavailable", "documents": [], "reason": "攻略检索暂不可用"}
