"""InsightRAG — Streamlit UI

Tier-1 features:
- Hybrid search (BM25 + Dense)
- Cross-encoder reranking
- Streaming answers
- Rich source display (snippets)
- Document management
"""

import os
from pathlib import Path
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

from src.config import SAMPLE_DIR, UPLOAD_DIR, TOP_K, HYBRID_WEIGHTS, USE_RERANKER
from src.document_loader import load_and_chunk
from src.vectorstore import get_vectorstore, add_documents, clear_vectorstore
from src.retriever import create_hybrid_retriever, retrieve_and_rerank
from src.rag_chain import get_llm, create_rag_chain, stream_answer
from src.evaluator import check_groundedness

load_dotenv()

st.set_page_config(
    page_title="InsightRAG",
    page_icon="💡",
    layout="wide",
)

st.title("💡 InsightRAG")
st.caption("Hybrid Retrieval + Cross-Encoder Reranking · Streaming Answers")

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    provider = st.selectbox(
        "LLM Provider",
        options=["ollama", "openai"],
        index=0,
        help="Ollama is fully local and free. OpenAI requires an API key.",
    )

    openai_key = None
    if provider == "openai":
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )

    st.divider()
    st.subheader("Retrieval Pipeline")
    st.markdown(
        f"""
- **Hybrid**: BM25 + Dense (weights {HYBRID_WEIGHTS[0]} / {HYBRID_WEIGHTS[1]})
- **Reranker**: `{'Enabled' if USE_RERANKER else 'Disabled'}` (Cross-Encoder)
        """
    )

    st.divider()
    st.subheader("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    use_sample = st.checkbox("Load sample company documents", value=True)

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Process Documents", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear All", use_container_width=True)

    if clear_btn:
        clear_vectorstore()
        for key in ["vectorstore", "retriever", "chunks", "doc_stats", "messages"]:
            st.session_state.pop(key, None)
        st.success("Index cleared.")
        st.rerun()

    # Document management panel
    if "doc_stats" in st.session_state and st.session_state["doc_stats"]:
        st.divider()
        st.subheader("Indexed Documents")
        total_chunks = 0
        for name, count in st.session_state["doc_stats"].items():
            st.markdown(f"- `{name}` — **{count}** chunks")
            total_chunks += count
        st.caption(f"Total chunks: {total_chunks}")

# --------------------------------------------------------------------------
# Process documents
# --------------------------------------------------------------------------
if process_btn:
    file_paths = []

    if use_sample and SAMPLE_DIR.exists():
        for f in SAMPLE_DIR.glob("*"):
            if f.suffix.lower() in {".txt", ".md", ".pdf"}:
                file_paths.append(f)

    if uploaded_files:
        for uf in uploaded_files:
            save_path = UPLOAD_DIR / uf.name
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            file_paths.append(save_path)

    if not file_paths:
        st.warning("No documents selected.")
    else:
        with st.spinner("Loading, chunking, embedding & building hybrid index..."):
            chunks = load_and_chunk(file_paths)
            vectorstore = get_vectorstore()
            add_documents(vectorstore, chunks)

            hybrid_retriever = create_hybrid_retriever(
                documents=chunks,
                vectorstore=vectorstore,
            )

            # Build document statistics
            stats = defaultdict(int)
            for c in chunks:
                stats[c.metadata.get("source", "unknown")] += 1

            st.session_state["vectorstore"] = vectorstore
            st.session_state["chunks"] = chunks
            st.session_state["retriever"] = hybrid_retriever
            st.session_state["doc_stats"] = dict(stats)

            st.success(
                f"Ready — {len(chunks)} chunks from {len(stats)} document(s). "
                f"Hybrid + Reranker enabled."
            )
            st.rerun()

# --------------------------------------------------------------------------
# Chat interface
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources & Snippets"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['name']}** (score: {src.get('score', 0):.3f})")
                    st.markdown(f"> {src['snippet']}")
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if "retriever" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("Please process some documents first using the sidebar.")
    else:
        with st.chat_message("assistant"):
            try:
                # 1. Retrieve + Rerank
                with st.spinner("Retrieving & reranking..."):
                    ranked = retrieve_and_rerank(
                        st.session_state["retriever"],
                        prompt,
                        top_k=TOP_K,
                    )
                    docs = [doc for doc, score in ranked]
                    scores = [score for doc, score in ranked]

                # 2. Stream the answer
                llm = get_llm(provider=provider, openai_api_key=openai_key or None)
                chain = create_rag_chain(llm)

                response_placeholder = st.empty()
                full_answer = ""

                for chunk in stream_answer(chain, prompt, docs):
                    full_answer += chunk
                    response_placeholder.markdown(full_answer + "▌")

                response_placeholder.markdown(full_answer)

                # 3. Rich source display
                sources_for_ui = []
                if docs:
                    with st.expander("Sources & Snippets", expanded=False):
                        for doc, score in zip(docs, scores):
                            name = doc.metadata.get("source", "unknown")
                            snippet = doc.page_content.strip().replace("\n", " ")
                            if len(snippet) > 280:
                                snippet = snippet[:280] + "..."
                            st.markdown(f"**{name}** · relevance: `{score:.3f}`")
                            st.markdown(f"> {snippet}")
                            st.markdown("---")
                            sources_for_ui.append(
                                {"name": name, "snippet": snippet, "score": float(score)}
                            )

                # 4. Groundedness indicator
                grounded = check_groundedness(full_answer, docs)
                st.caption(
                    f"Pipeline: Hybrid → Reranker · "
                    f"Groundedness: {grounded['score']} · "
                    f"{'Looks grounded' if grounded['is_grounded'] else 'May be weakly grounded'}"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_answer,
                        "sources": sources_for_ui,
                    }
                )

            except Exception as e:
                st.error(
                    f"Error: {e}\n\n"
                    "If using Ollama, make sure it is running (`ollama serve`) "
                    "and the model is pulled (`ollama pull llama3.2`)."
                )
