"""Guide retriever using FAISS vector search."""

from __future__ import annotations

import logging
from typing import Any

from gamecoach.rag.indexer import get_or_build_index

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.3


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Semantic search over guide documents."""
    try:
        vectorstore = get_or_build_index()
    except Exception:
        logger.exception("Failed to get vector index")
        return []

    try:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k * 2)
    except Exception:
        logger.exception("Vector search failed")
        return []

    results = []
    for doc, score in docs_with_scores:
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

    logger.info("RAG query '%s' -> %d results", query[:40], len(results))
    return results
