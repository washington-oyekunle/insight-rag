"""Load and chunk documents for RAG."""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(file_paths: List[str | Path]) -> List[Document]:
    """Load PDF and TXT files into LangChain Documents."""
    documents = []

    for path in file_paths:
        path = Path(path)
        if not path.exists():
            continue

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()
        elif path.suffix.lower() in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
        else:
            continue

        # Add source metadata
        for doc in docs:
            doc.metadata["source"] = path.name

        documents.extend(docs)

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def load_and_chunk(file_paths: List[str | Path]) -> List[Document]:
    """Convenience: load + chunk in one call."""
    docs = load_documents(file_paths)
    return chunk_documents(docs)
