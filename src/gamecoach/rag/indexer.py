"""FAISS 索引构建与管理。

将攻略文档切分、向量化并持久化到 data/vector_store/。
支持增量加载：如果已有索引则从磁盘加载，否则新建。
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from gamecoach.config.llm import get_embeddings
from gamecoach.rag.loader import load_guides

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = Path(__file__).resolve().parents[3] / "data" / "vector_store"
INDEX_FILE = VECTOR_STORE_DIR / "faiss_index"
DOCSTORE_FILE = VECTOR_STORE_DIR / "faiss_docstore.pkl"

# 文本切分器
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=128,
    separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ".", "。", "；", "，"],
    keep_separator=True,
)

_vectorstore: FAISS | None = None


def _split_documents(docs: list[Document]) -> list[Document]:
    """切分文档为小块。"""
    chunks = TEXT_SPLITTER.split_documents(docs)
    logger.info("文档切分为 %d 个 chunk", len(chunks))
    return chunks


def _build_new_index(force: bool = False) -> FAISS:
    """从零构建 FAISS 索引并持久化。"""
    docs = load_guides()
    if not docs:
        logger.warning("没有可索引的攻略文档")
        raise RuntimeError("攻略文档目录为空，无法构建索引")

    chunks = _split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 持久化
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_STORE_DIR))
    logger.info("FAISS 索引已保存到 %s (%d 向量)", VECTOR_STORE_DIR, len(chunks))

    return vectorstore


def _load_existing_index() -> FAISS | None:
    """尝试从磁盘加载已有索引。"""
    if not INDEX_FILE.exists():
        return None
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            str(VECTOR_STORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("从磁盘加载了已有 FAISS 索引")
        return vectorstore
    except Exception:
        logger.exception("加载已有索引失败，将重建")
        return None


def get_or_build_index(force_rebuild: bool = False) -> FAISS:
    """获取或构建 FAISS 向量索引。

    优先从磁盘加载持久化索引，如果不存在或 force_rebuild=True 则重建。

    Args:
        force_rebuild: 强制重建索引。

    Returns:
        FAISS 向量存储实例。
    """
    global _vectorstore

    if _vectorstore is not None and not force_rebuild:
        return _vectorstore

    if not force_rebuild:
        _vectorstore = _load_existing_index()

    if _vectorstore is None:
        _vectorstore = _build_new_index(force=force_rebuild)

    return _vectorstore
