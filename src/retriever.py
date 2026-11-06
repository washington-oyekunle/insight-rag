"""Hybrid retrieval: BM25 (sparse) + Dense vectors (Chroma).

Uses LangChain EnsembleRetriever with Reciprocal Rank Fusion under the hood.
This significantly improves retrieval quality over pure vector search,
especially for keyword-heavy or exact-match queries.
"""

from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import TOP_K, DENSE_K, BM25_K, HYBRID_WEIGHTS
from src.vectorstore import get_vectorstore


def create_bm25_retriever(documents: List[Document], k: int = BM25_K) -> BM25Retriever:
    """Create a BM25 sparse retriever from a list of documents."""
    if not documents:
        raise ValueError("Cannot create BM25 retriever with empty document list")

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def create_dense_retriever(vectorstore, k: int = DENSE_K):
    """Create a dense (vector) retriever from a Chroma vectorstore."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def create_hybrid_retriever(
    documents: List[Document],
    vectorstore=None,
    weights: Optional[List[float]] = None,
    top_k: int = TOP_K,
) -> BaseRetriever:
    """Create a hybrid retriever combining BM25 + dense vectors.

    Args:
        documents: The chunked documents (needed for BM25).
        vectorstore: Existing Chroma store. Created if None.
        weights: [bm25_weight, dense_weight]. Defaults to config HYBRID_WEIGHTS.
        top_k: Final number of documents returned after fusion.

    Returns:
        An EnsembleRetriever that fuses results via reciprocal rank fusion.
    """
    if weights is None:
        weights = HYBRID_WEIGHTS

    if vectorstore is None:
        vectorstore = get_vectorstore()

    bm25_retriever = create_bm25_retriever(documents, k=BM25_K)
    dense_retriever = create_dense_retriever(vectorstore, k=DENSE_K)

    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=weights,
    )

    # EnsembleRetriever returns up to (bm25_k + dense_k) candidates
    # then ranks them. We keep the fused top results via search_kwargs if needed.
    # For simplicity we let it return the fused list and slice later if required.
    return hybrid


def retrieve_with_scores(retriever: BaseRetriever, query: str, k: int = TOP_K) -> List[Document]:
    """Convenience helper that returns the top-k documents."""
    docs = retriever.invoke(query)
    return docs[:k]
