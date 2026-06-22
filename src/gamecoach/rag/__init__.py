"""RAG 模块 — FAISS 向量检索 + DashScope Embedding。

提供攻略文档的加载（loader）、索引（indexer）和检索（retriever）能力。
索引首次构建后持久化到 data/vector_store/，后续秒级加载。
"""

from gamecoach.rag.retriever import retrieve

__all__ = ["retrieve"]
