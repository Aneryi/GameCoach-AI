"""
攻略检索器。

提供语义检索接口，通过 FAISS 向量相似度搜索返回最相关的攻略片段。

检索流程：
1. get_or_build_index() → 加载/构建 FAISS 向量索引
2. similarity_search_with_score() → 返回 (Document, L2距离) 列表
3. L2距离 → 相似度分数转换: 1/(1+score)
4. 按 relevance_threshold 过滤低分结果
5. 截取 snippet（前300字符），返回 top_k 条结果
"""

from __future__ import annotations

import logging
from typing import Any

from gamecoach.rag.indexer import get_or_build_index

logger = logging.getLogger(__name__)

# 相关性阈值：低于此 L2→相似度转换分数的结果被丢弃
# 0.3 是经验值——低于 0.3 的 snippet 和查询基本无关
RELEVANCE_THRESHOLD = 0.3


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    检索与查询最相关的攻略片段。

    Args:
        query: 自然语言查询，如 "团队站位技巧"。
        top_k: 返回结果数量上限，默认 5。

    Returns:
        文档列表，每项包含 source（文件名）、title（标题）、
        score（0-1 相似度）、snippet（前300字符摘要）。
        如果索引构建失败或检索异常，返回空列表。
    """
    # 获取 FAISS 索引
    try:
        vectorstore = get_or_build_index()
    except Exception:
        logger.exception("无法获取向量索引")
        return []

    # 相似度检索（返回 L2 距离）
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k * 2)
    except Exception:
        logger.exception("向量检索失败")
        return []

    # 转换 L2 距离 → 0~1 相似度，过滤低分
    results = []
    for doc, score in docs_with_scores:
        # FAISS 返回 L2 距离（越小越相似）→ 转为相似度分数
        similarity = 1.0 / (1.0 + score)
        if similarity < RELEVANCE_THRESHOLD:
            continue

        snippet = doc.page_content[:300].replace("\n", " ").strip()
        results.append({
            "source": doc.metadata.get("source", "unknown"),
            "title": doc.metadata.get("title", "Untitled"),
            "score": round(similarity, 4),
            "snippet": snippet,
        })

        if len(results) >= top_k:
            break

    logger.info("RAG 检索 '%s' → %d 条结果", query[:40], len(results))
    return results
