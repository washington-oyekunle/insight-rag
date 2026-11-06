"""Embedding model setup."""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a local HuggingFace embedding model.

    Uses all-MiniLM-L6-v2 — fast, free, and strong for semantic search.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
    )
