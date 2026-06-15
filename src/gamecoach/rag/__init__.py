"""GameCoach AI RAG 模块。

提供攻略文档的加载、索引和检索能力。
使用 FAISS 向量库 + 本地 embedding 做语义检索。
"""

from gamecoach.rag.retriever import retrieve

__all__ = ["retrieve"]
