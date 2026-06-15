"""攻略检索器。

提供语义检索接口，根据查询返回最相关的攻略片段。
"""

from __future__ import annotations

import logging
from typing import Any

from gamecoach.rag.indexer import get_or_build_index

logger = logging.getLogger(__name__)

# 相关性阈值（低于此分数的结果被过滤）
RELEVANCE_THRESHOLD = 0.3


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """检索与查询最相关的攻略片段。

    Args:
        query: 自然语言查询，如 "射手团战站位技巧"。
        top_k: 返回结果数量上限。

    Returns:
        文档列表，每项包含 source, title, score, snippet。
    """
    try:
        vectorstore = get_or_build_index()
    except Exception:
        logger.exception("无法获取向量索引")
        return []

    try:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k * 2)
    except Exception:
        logger.exception("向量检索失败")
        return []

    results = []
    for doc, score in docs_with_scores:
        # FAISS 返回的是 L2 距离，越小越相似
        # 转为 0-1 相似度分数
        similarity = 1.0 / (1.0 + score)
        if similarity < RELEVANCE_THRESHOLD:
            continue

        snippet = doc.page_content[:300].replace("\n", " ").strip()
        results.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "title": doc.metadata.get("title", "无标题"),
                "score": round(similarity, 4),
                "snippet": snippet,
            }
        )

        if len(results) >= top_k:
            break

    logger.info("RAG 检索 '%s' → %d 条结果", query[:40], len(results))
    return results
