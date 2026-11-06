"""Chroma vector store operations."""

from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import CHROMA_DIR
from src.embeddings import get_embeddings


def get_vectorstore(
    embeddings: Optional[Embeddings] = None,
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = "insight_rag",
) -> Chroma:
    """Get or create a persistent Chroma vector store."""
    if embeddings is None:
        embeddings = get_embeddings()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def add_documents(vectorstore: Chroma, documents: List[Document]) -> None:
    """Add documents to the vector store."""
    if documents:
        vectorstore.add_documents(documents)


def clear_vectorstore(persist_directory: str | Path = CHROMA_DIR) -> None:
    """Delete the persistent Chroma directory (use with care)."""
    import shutil

    path = Path(persist_directory)
    if path.exists():
        shutil.rmtree(path)
