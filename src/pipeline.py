"""High-level RAG pipeline used by both Streamlit and FastAPI."""

from typing import List, Optional, Tuple, Iterator, Dict, Any
from collections import defaultdict

from langchain_core.documents import Document

from src.config import SAMPLE_DIR, TOP_K, USE_QUERY_REWRITE
from src.document_loader import load_and_chunk
from src.vectorstore import get_vectorstore, add_documents, clear_vectorstore
from src.retriever import create_hybrid_retriever, retrieve_and_rerank
from src.query_rewrite import rewrite_query
from src.rag_chain import get_llm, create_rag_chain, stream_answer, answer_question
from src.evaluator import check_groundedness


class RAGPipeline:
    """Stateful RAG pipeline (documents + retriever)."""

    def __init__(self):
        self.chunks: List[Document] = []
        self.vectorstore = None
        self.retriever = None
        self.doc_stats: Dict[str, int] = {}

    @property
    def is_ready(self) -> bool:
        return self.retriever is not None and len(self.chunks) > 0

    def index_files(self, file_paths: List) -> Dict[str, Any]:
        chunks = load_and_chunk(file_paths)
        if not chunks:
            return {"chunks": 0, "documents": 0}

        self.vectorstore = get_vectorstore()
        add_documents(self.vectorstore, chunks)
        self.chunks = chunks
        self.retriever = create_hybrid_retriever(
            documents=chunks,
            vectorstore=self.vectorstore,
        )

        stats = defaultdict(int)
        for c in chunks:
            stats[c.metadata.get("source", "unknown")] += 1
        self.doc_stats = dict(stats)

        return {"chunks": len(chunks), "documents": len(stats), "doc_stats": self.doc_stats}

    def index_sample(self) -> Dict[str, Any]:
        files = [f for f in SAMPLE_DIR.glob("*") if f.suffix.lower() in {".txt", ".md", ".pdf"}]
        return self.index_files(files)

    def clear(self) -> None:
        clear_vectorstore()
        self.chunks = []
        self.vectorstore = None
        self.retriever = None
        self.doc_stats = {}

    def query(
        self,
        question: str,
        provider: str = "ollama",
        openai_api_key: Optional[str] = None,
        source_filter: Optional[List[str]] = None,
        use_rewrite: bool = USE_QUERY_REWRITE,
        stream: bool = False,
    ):
        if not self.is_ready:
            raise RuntimeError("No documents indexed. Call index_files first.")

        llm = get_llm(provider=provider, openai_api_key=openai_api_key)

        # Query rewriting
        search_query = rewrite_query(question, llm, enabled=use_rewrite)

        # Retrieve + rerank
        ranked = retrieve_and_rerank(
            self.retriever,
            search_query,
            top_k=TOP_K,
            source_filter=source_filter,
        )
        docs = [doc for doc, score in ranked]
        scores = [float(score) for doc, score in ranked]

        chain = create_rag_chain(llm)

        sources = []
        for doc, score in zip(docs, scores):
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:280] + "..."
            sources.append(
                {
                    "name": doc.metadata.get("source", "unknown"),
                    "snippet": snippet,
                    "score": score,
                }
            )

        if stream:
            def generator() -> Iterator[str]:
                for chunk in stream_answer(chain, question, docs):
                    yield chunk

            return {
                "search_query": search_query,
                "sources": sources,
                "docs": docs,
                "stream": generator(),
            }

        answer = answer_question(chain, question, docs)
        grounded = check_groundedness(answer, docs)

        return {
            "answer": answer,
            "search_query": search_query,
            "sources": sources,
            "groundedness": grounded,
        }


# Singleton used by the API
pipeline = RAGPipeline()
