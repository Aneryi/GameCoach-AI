"""攻略文档加载器。

从 data/guides/ 目录加载所有 .md 文件，
提取标题和元数据，返回 LangChain Document 列表。
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

GUIDES_DIR = Path(__file__).resolve().parents[3] / "data" / "guides"


def load_guides() -> list[Document]:
    """加载 data/guides/ 下所有 .md 攻略文档。

    Returns:
        LangChain Document 列表，每个 Document 的 metadata 包含 source 和 title。
    """
    if not GUIDES_DIR.exists():
        logger.warning("攻略目录不存在: %s", GUIDES_DIR)
        return []

    loader = DirectoryLoader(
        str(GUIDES_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()

    if not docs:
        logger.warning("攻略目录中没有找到 .md 文件")
        return docs

    # 从内容中提取标题作为 metadata
    for doc in docs:
        source = Path(doc.metadata.get("source", ""))
        doc.metadata["source"] = source.name

        # 提取第一个 # 标题
        for line in doc.page_content.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                doc.metadata["title"] = line[2:].strip()
                break
        if "title" not in doc.metadata:
            doc.metadata["title"] = source.stem

    logger.info("加载了 %d 篇攻略文档", len(docs))
    return docs
