"""Hybrid retrieval + Cross-Encoder reranking + optional metadata filtering.

Pipeline:
1. (Optional) Query rewriting
2. BM25 + Dense vectors via EnsembleRetriever
3. Optional metadata filter (e.g. by source)
4. Cross-encoder reranker
"""

from typing import List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import (
    TOP_K,
    DENSE_K,
    BM25_K,
    HYBRID_WEIGHTS,
    RERANK_CANDIDATES,
    RERANKER_MODEL,
    USE_RERANKER,
)
from src.vectorstore import get_vectorstore

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def create_bm25_retriever(documents: List[Document], k: int = BM25_K) -> BM25Retriever:
    if not documents:
        raise ValueError("Cannot create BM25 retriever with empty document list")
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def create_dense_retriever(vectorstore, k: int = DENSE_K):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def create_hybrid_retriever(
    documents: List[Document],
    vectorstore=None,
    weights: Optional[List[float]] = None,
) -> BaseRetriever:
    if weights is None:
        weights = HYBRID_WEIGHTS
    if vectorstore is None:
        vectorstore = get_vectorstore()

    bm25_retriever = create_bm25_retriever(documents, k=BM25_K)
    dense_retriever = create_dense_retriever(vectorstore, k=DENSE_K)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=weights,
    )


def filter_by_sources(
    documents: List[Document],
    allowed_sources: Optional[List[str]] = None,
) -> List[Document]:
    """Filter documents by source filename metadata."""
    if not allowed_sources:
        return documents
    allowed = set(allowed_sources)
    return [d for d in documents if d.metadata.get("source") in allowed]


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = TOP_K,
) -> List[Tuple[Document, float]]:
    if not documents:
        return []

    if not USE_RERANKER:
        return [(doc, 0.0) for doc in documents[:top_k]]

    model = _get_reranker()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def retrieve_and_rerank(
    retriever: BaseRetriever,
    query: str,
    top_k: int = TOP_K,
    candidate_k: int = RERANK_CANDIDATES,
    source_filter: Optional[List[str]] = None,
) -> List[Tuple[Document, float]]:
    """Hybrid retrieve → optional source filter → cross-encoder rerank."""
    candidates = retriever.invoke(query)[:candidate_k]
    candidates = filter_by_sources(candidates, source_filter)
    return rerank_documents(query, candidates, top_k=top_k)
