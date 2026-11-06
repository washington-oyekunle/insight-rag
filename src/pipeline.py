"""High-level RAG pipeline used by both Streamlit and FastAPI.

Tier-3 capabilities:
- Query rewriting
- Agentic routing (retrieve / clarify / chitchat)
- Self-correcting retrieval loop
- Structured observability logs
"""

from typing import List, Optional, Iterator, Dict, Any
from collections import defaultdict

from langchain_core.documents import Document

from src.config import (
    SAMPLE_DIR,
    TOP_K,
    USE_QUERY_REWRITE,
    USE_SELF_CORRECTION,
    MAX_CORRECTION_ROUNDS,
    GROUNDEDNESS_THRESHOLD,
)
from src.document_loader import load_and_chunk
from src.vectorstore import get_vectorstore, add_documents, clear_vectorstore
from src.retriever import create_hybrid_retriever, retrieve_and_rerank
from src.query_rewrite import rewrite_query
from src.rag_chain import get_llm, create_rag_chain, stream_answer, answer_question
from src.evaluator import check_groundedness
from src.agent import route_question, reformulate_query, needs_correction
from src.observability import Timer, log_query


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

    def _build_sources(self, docs: List[Document], scores: List[float]) -> List[dict]:
        sources = []
        for doc, score in zip(docs, scores):
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:280] + "..."
            sources.append(
                {
                    "name": doc.metadata.get("source", "unknown"),
                    "snippet": snippet,
                    "score": float(score),
                }
            )
        return sources

    def _retrieve(
        self,
        search_query: str,
        source_filter: Optional[List[str]] = None,
    ):
        ranked = retrieve_and_rerank(
            self.retriever,
            search_query,
            top_k=TOP_K,
            source_filter=source_filter,
        )
        docs = [doc for doc, score in ranked]
        scores = [float(score) for doc, score in ranked]
        return docs, scores

    def query(
        self,
        question: str,
        provider: str = "ollama",
        openai_api_key: Optional[str] = None,
        source_filter: Optional[List[str]] = None,
        use_rewrite: bool = USE_QUERY_REWRITE,
        use_self_correction: bool = USE_SELF_CORRECTION,
        stream: bool = False,
    ):
        timer = Timer()

        llm = get_llm(provider=provider, openai_api_key=openai_api_key)

        # --- Agentic routing ---
        route = route_question(question, llm)

        if route == "CHITCHAT":
            answer = (
                "Hello! I can answer questions about your indexed documents. "
                "Upload or load documents, then ask me anything about them."
            )
            result = {
                "answer": answer,
                "search_query": "",
                "sources": [],
                "groundedness": {"score": 1.0, "is_grounded": True, "note": "chitchat"},
                "route": route,
                "self_corrected": False,
                "correction_rounds": 0,
            }
            log_query(
                {
                    "question": question,
                    "route": route,
                    "latency_ms": timer.ms(),
                    "self_corrected": False,
                    "groundedness": result["groundedness"],
                }
            )
            if stream:
                def gen():
                    yield answer
                result["stream"] = gen()
                result["docs"] = []
            return result

        if route == "CLARIFY":
            answer = (
                "I need a bit more detail to search effectively. "
                "Could you specify which topic, policy, or document area you mean?"
            )
            result = {
                "answer": answer,
                "search_query": "",
                "sources": [],
                "groundedness": {"score": 1.0, "is_grounded": True, "note": "clarify"},
                "route": route,
                "self_corrected": False,
                "correction_rounds": 0,
            }
            log_query(
                {
                    "question": question,
                    "route": route,
                    "latency_ms": timer.ms(),
                    "self_corrected": False,
                    "groundedness": result["groundedness"],
                }
            )
            if stream:
                def gen():
                    yield answer
                result["stream"] = gen()
                result["docs"] = []
            return result

        # --- Retrieval path ---
        if not self.is_ready:
            raise RuntimeError("No documents indexed. Call index_files first.")

        search_query = rewrite_query(question, llm, enabled=use_rewrite)
        docs, scores = self._retrieve(search_query, source_filter)
        chain = create_rag_chain(llm)

        # Non-stream path supports full self-correction
        if not stream:
            answer = answer_question(chain, question, docs)
            grounded = check_groundedness(answer, docs)
            corrected = False
            rounds = 0

            while (
                use_self_correction
                and rounds < MAX_CORRECTION_ROUNDS
                and needs_correction(grounded, GROUNDEDNESS_THRESHOLD)
            ):
                rounds += 1
                search_query = reformulate_query(question, search_query, llm)
                docs, scores = self._retrieve(search_query, source_filter)
                answer = answer_question(chain, question, docs)
                grounded = check_groundedness(answer, docs)
                corrected = True

            sources = self._build_sources(docs, scores)
            result = {
                "answer": answer,
                "search_query": search_query,
                "sources": sources,
                "groundedness": grounded,
                "route": "RETRIEVE",
                "self_corrected": corrected,
                "correction_rounds": rounds,
            }

            log_query(
                {
                    "question": question,
                    "search_query": search_query,
                    "route": "RETRIEVE",
                    "latency_ms": timer.ms(),
                    "self_corrected": corrected,
                    "correction_rounds": rounds,
                    "num_sources": len(sources),
                    "source_names": [s["name"] for s in sources],
                    "groundedness": grounded,
                }
            )
            return result

        # Stream path: retrieve once (correction requires full answer first)
        sources = self._build_sources(docs, scores)

        def generator() -> Iterator[str]:
            for chunk in stream_answer(chain, question, docs):
                yield chunk

        result = {
            "search_query": search_query,
            "sources": sources,
            "docs": docs,
            "stream": generator(),
            "route": "RETRIEVE",
            "self_corrected": False,
            "correction_rounds": 0,
        }

        log_query(
            {
                "question": question,
                "search_query": search_query,
                "route": "RETRIEVE",
                "latency_ms": timer.ms(),
                "self_corrected": False,
                "num_sources": len(sources),
                "source_names": [s["name"] for s in sources],
                "stream": True,
            }
        )
        return result


# Singleton used by the API
pipeline = RAGPipeline()
