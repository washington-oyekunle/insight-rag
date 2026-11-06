"""InsightRAG — Streamlit UI with Hybrid Search"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.config import SAMPLE_DIR, UPLOAD_DIR, TOP_K, HYBRID_WEIGHTS
from src.document_loader import load_and_chunk
from src.vectorstore import get_vectorstore, add_documents, clear_vectorstore
from src.retriever import create_hybrid_retriever, retrieve_with_scores
from src.rag_chain import get_llm, create_rag_chain, answer_question
from src.evaluator import check_groundedness

load_dotenv()

st.set_page_config(
    page_title="InsightRAG",
    page_icon="💡",
    layout="wide",
)

st.title("💡 InsightRAG")
st.caption("Intelligent Document Q&A · Hybrid Retrieval (BM25 + Dense Vectors)")

# Sidebar
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
    st.subheader("Retrieval")
    st.markdown(
        f"**Mode:** Hybrid (BM25 + Dense)  \n"
        f"Weights → BM25: `{HYBRID_WEIGHTS[0]}` · Dense: `{HYBRID_WEIGHTS[1]}`"
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
        clear_btn = st.button("Clear Index", use_container_width=True)

    if clear_btn:
        clear_vectorstore()
        for key in ["vectorstore", "retriever", "chunks", "messages"]:
            st.session_state.pop(key, None)
        st.success("Index cleared.")
        st.rerun()

# Process documents
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

            # Build hybrid retriever (BM25 needs the documents in memory)
            hybrid_retriever = create_hybrid_retriever(
                documents=chunks,
                vectorstore=vectorstore,
            )

            st.session_state["vectorstore"] = vectorstore
            st.session_state["chunks"] = chunks
            st.session_state["retriever"] = hybrid_retriever

            st.success(
                f"Ready — {len(chunks)} chunks from {len(file_paths)} file(s). "
                f"Hybrid search enabled."
            )

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

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
            with st.spinner("Retrieving (hybrid) + generating answer..."):
                try:
                    llm = get_llm(provider=provider, openai_api_key=openai_key or None)
                    chain = create_rag_chain(st.session_state["retriever"], llm)

                    # Retrieve docs (for sources + evaluation)
                    docs = retrieve_with_scores(
                        st.session_state["retriever"], prompt, k=TOP_K
                    )
                    answer = answer_question(chain, prompt)

                    st.markdown(answer)

                    sources = list({d.metadata.get("source", "unknown") for d in docs})
                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                st.markdown(f"- `{s}`")

                    grounded = check_groundedness(answer, docs)
                    st.caption(
                        f"Retrieval: Hybrid (BM25 + Dense) · "
                        f"Groundedness: {grounded['score']} · "
                        f"{'Looks grounded' if grounded['is_grounded'] else 'May be weakly grounded'}"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except Exception as e:
                    st.error(
                        f"Error: {e}\n\n"
                        "If using Ollama, make sure it is running (`ollama serve`) "
                        "and the model is pulled (`ollama pull llama3.2`)."
                    )
