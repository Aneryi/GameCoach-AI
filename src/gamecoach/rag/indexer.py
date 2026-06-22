"""
FAISS 索引构建与管理。

将攻略文档切分为语义块，通过 Embedding 模型向量化，
存入 FAISS 向量库并持久化到磁盘（data/vector_store/）。

索引由 indexer.loader 加载的数据中。格式匹配文档存储，
文档由 retriever 修改。格式匹配文档存储索引修改前：
- 首次运行：加载 data/guides/ 下所有 .md → 切分 → Embedding → 构建 FAISS → 持久化
- 后续运行：从 data/vector_store/ 加载已有索引（秒级加载，跳过 Embedding）
- force_rebuild=True：强制重建（攻略文档更新后使用）
"""

from __future__ import annotations

import logging
from pathlib import Path

# RecursiveCharacterTextSplitter: 按分隔符优先级递归切分文档
# 分离器：每个字符都有默认自然层级递切（## > ### > \n > 空格 > 句号），避免暴力切碎内容
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS: Meta 开源的高效向量相似度搜索库
# from_documents(docs, embeddings): 将文档列表向量化后构建索引
# save_local(path): 将索引持久化到磁盘
# load_local(path, embeddings): 从磁盘加载已有索引
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from gamecoach.config.llm import get_embeddings
from gamecoach.rag.loader import load_guides

logger = logging.getLogger(__name__)

# 索引持久化路径
VECTOR_STORE_DIR = Path(__file__).resolve().parents[3] / "data" / "vector_store"

# 文本切分配置：
# chunk_size=800: 每个块约 800 字符（约 400-500 tokens），适合攻略段落
# chunk_overlap=128: 块间重叠 128 字符，避免关键信息被切断在块边界
# separators: 按优先级从高到低的切分分隔符
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=128,
    separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ".", "。", "；", "，"],
    keep_separator=True,  # 保留分隔符，维持 markdown 标题信息
)

# 模块级向量库缓存 —— 生命周期内只加载一次
_vectorstore: FAISS | None = None


def _split_documents(docs: list[Document]) -> list[Document]:
    """将加载的文档按配置切分为小块。"""
    chunks = TEXT_SPLITTER.split_documents(docs)
    logger.info("文档切分为 %d 个 chunk", len(chunks))
    return chunks


def _build_new_index() -> FAISS:
    """
    从零构建 FAISS 索引。

    流程：加载攻略文档 → 切分 → Embedding 向量化 → 构建 FAISS 索引 → 持久化。
    如果 DashScope API Key 未配置（get_embeddings() 返回 None），抛 RuntimeError。
    """
    docs = load_guides()
    if not docs:
        raise RuntimeError("攻略文档目录为空，无法构建索引")

    chunks = _split_documents(docs)
    embeddings = get_embeddings()
    if embeddings is None:
        raise RuntimeError("Embedding 模型不可用（DASHSCOPE_API_KEY 未配置）")

    # from_documents: 自动调用 embeddings.embed_documents 向量化每个 chunk
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 持久化到 data/vector_store/
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_STORE_DIR))
    logger.info("FAISS 索引已保存到 %s (%d 向量)", VECTOR_STORE_DIR, len(chunks))

    return vectorstore


def _load_existing_index() -> FAISS | None:
    """尝试从磁盘加载已有索引。索引文件不存在则返回 None。"""
    if not (VECTOR_STORE_DIR / "index.faiss").exists():
        return None
    try:
        embeddings = get_embeddings()
        if embeddings is None:
            return None
        # allow_dangerous_deserialization=True: FAISS 索引文件是 pickle 序列化的，
        # LangChain 要求显式确认安全风险（索引文件是本地生成的，可信任）
        vectorstore = FAISS.load_local(
            str(VECTOR_STORE_DIR), embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("从磁盘加载了已有 FAISS 索引")
        return vectorstore
    except Exception:
        logger.exception("加载已有索引失败，将重建")
        return None


def get_or_build_index(force_rebuild: bool = False) -> FAISS:
    """
    获取或构建 FAISS 向量索引（全局单例）。

    加载策略（按优先级）：
    1. 返回内存缓存（_vectorstore）—— 已有就复用
    2. 从磁盘加载已有索引 —— 第二次运行时秒级加载
    3. 从零构建新索引 —— 首次运行或 force_rebuild=True 时

    Args:
        force_rebuild: 强制重建索引（攻略文档更新后使用）。

    Returns:
        FAISS 向量库实例。
    """
    global _vectorstore

    if _vectorstore is not None and not force_rebuild:
        return _vectorstore

    if not force_rebuild:
        _vectorstore = _load_existing_index()

    if _vectorstore is None:
        _vectorstore = _build_new_index()

    return _vectorstore
