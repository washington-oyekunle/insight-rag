"""InsightRAG — Streamlit UI (Tier-2)

Features:
- Hybrid search + Cross-encoder reranking
- Query rewriting
- Streaming answers
- Metadata (source) filtering
- Document management
- Chat history persistence
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.config import SAMPLE_DIR, UPLOAD_DIR, TOP_K, HYBRID_WEIGHTS, USE_RERANKER, USE_QUERY_REWRITE
from src.pipeline import RAGPipeline
from src.chat_history import create_session, append_message, load_history

load_dotenv()

st.set_page_config(page_title="InsightRAG", page_icon="💡", layout="wide")

st.title("💡 InsightRAG")
st.caption("Hybrid + Reranker + Query Rewrite · Streaming · Source Filtering")

# --------------------------------------------------------------------------
# Session-scoped pipeline
# --------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "session_id" not in st.session_state:
    st.session_state.session_id = create_session()

if "messages" not in st.session_state:
    st.session_state.messages = load_history(st.session_state.session_id)

pipeline: RAGPipeline = st.session_state.pipeline

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    provider = st.selectbox("LLM Provider", ["ollama", "openai"], index=0)
    openai_key = None
    if provider == "openai":
        openai_key = st.text_input(
            "OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", "")
        )

    use_rewrite = st.checkbox("Query rewriting", value=USE_QUERY_REWRITE)

    st.divider()
    st.subheader("Retrieval Pipeline")
    st.markdown(
        f"""
- Hybrid: BM25 + Dense (`{HYBRID_WEIGHTS[0]}` / `{HYBRID_WEIGHTS[1]}`)
- Reranker: {'On' if USE_RERANKER else 'Off'}
- Query rewrite: {'On' if use_rewrite else 'Off'}
        """
    )

    st.divider()
    st.subheader("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    use_sample = st.checkbox("Load sample documents", value=True)

    # Source filter (metadata filtering)
    available_sources = list(pipeline.doc_stats.keys()) if pipeline.doc_stats else []
    source_filter = None
    if available_sources:
        selected = st.multiselect(
            "Filter by document (optional)",
            options=available_sources,
            default=available_sources,
            help="Only retrieve from selected documents",
        )
        if selected and set(selected) != set(available_sources):
            source_filter = selected

    c1, c2 = st.columns(2)
    with c1:
        process_btn = st.button("Process", type="primary", use_container_width=True)
    with c2:
        clear_btn = st.button("Clear All", use_container_width=True)

    if clear_btn:
        pipeline.clear()
        st.session_state.messages = []
        st.session_state.session_id = create_session()
        st.success("Index cleared.")
        st.rerun()

    if pipeline.doc_stats:
        st.divider()
        st.subheader("Indexed Documents")
        for name, count in pipeline.doc_stats.items():
            st.markdown(f"- `{name}` — **{count}** chunks")
        st.caption(f"Total chunks: {len(pipeline.chunks)}")

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id}`")

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
        with st.spinner("Indexing documents..."):
            result = pipeline.index_files(file_paths)
            st.success(
                f"Ready — {result['chunks']} chunks from {result['documents']} document(s)."
            )
            st.rerun()

# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources & Snippets"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['name']}** (score: {src.get('score', 0):.3f})")
                    st.markdown(f"> {src['snippet']}")
                    st.markdown("---")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    append_message(st.session_state.session_id, "user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    if not pipeline.is_ready:
        with st.chat_message("assistant"):
            st.warning("Please process some documents first.")
    else:
        with st.chat_message("assistant"):
            try:
                result = pipeline.query(
                    question=prompt,
                    provider=provider,
                    openai_api_key=openai_key or None,
                    source_filter=source_filter,
                    use_rewrite=use_rewrite,
                    stream=True,
                )

                if result["search_query"] != prompt:
                    st.caption(f"Rewritten query: _{result['search_query']}_")

                response_placeholder = st.empty()
                full_answer = ""
                for chunk in result["stream"]:
                    full_answer += chunk
                    response_placeholder.markdown(full_answer + "▌")
                response_placeholder.markdown(full_answer)

                sources = result["sources"]
                if sources:
                    with st.expander("Sources & Snippets"):
                        for src in sources:
                            st.markdown(
                                f"**{src['name']}** · relevance: `{src['score']:.3f}`"
                            )
                            st.markdown(f"> {src['snippet']}")
                            st.markdown("---")

                st.caption("Pipeline: Rewrite → Hybrid → Rerank → Generate")

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_answer, "sources": sources}
                )
                append_message(
                    st.session_state.session_id,
                    "assistant",
                    full_answer,
                    sources=sources,
                )

            except Exception as e:
                st.error(
                    f"Error: {e}\n\n"
                    "If using Ollama, ensure it is running and the model is pulled."
                )
