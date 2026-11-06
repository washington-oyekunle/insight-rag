# InsightRAG 💡

**Intelligent Document Q&A with Hybrid Retrieval (BM25 + Dense Vectors)**

A clean, modular, production-style RAG system built as a strong portfolio project for AI / ML / LLM Engineer roles.

Upload documents → Ask natural language questions → Get grounded answers with source citations.

> Now features **Hybrid Search**: combines classic keyword search (BM25) with semantic vector search for significantly better retrieval quality.

---

## Why This Project Helps You Get Hired

Hiring managers for Applied AI roles look for more than notebooks. This project demonstrates:

| Skill | How it's shown |
|-------|----------------|
| **Hybrid Retrieval** | BM25 (sparse) + Dense embeddings fused via EnsembleRetriever |
| **RAG Architecture** | Full pipeline: load → chunk → embed → retrieve → generate |
| **Vector Databases** | Persistent ChromaDB |
| **Embeddings** | Local `sentence-transformers/all-MiniLM-L6-v2` |
| **LLM Integration** | Ollama (local/free) + OpenAI support |
| **Evaluation Mindset** | Groundedness heuristics + clear extension path to RAGAS |
| **Software Design** | Modular `src/`, config-driven, clean separation of concerns |
| **Product Sense** | Working Streamlit chat UI with sources |

---

## Architecture

```
Documents (PDF / TXT)
        ↓
   Load + Chunk (RecursiveCharacterTextSplitter)
        ↓
   ┌──────────────────┐
   │                     │
 BM25 Index          Dense Embeddings
 (keyword)           (all-MiniLM-L6-v2)
   │                     │
   └──────────────────┘
                ↓
     EnsembleRetriever (Reciprocal Rank Fusion)
                ↓
     Top-k relevant chunks + sources
                ↓
     Prompt + Context → LLM → Answer
```

**Why Hybrid Search?**  
Pure vector search can miss exact keywords or rare terms. BM25 is excellent at keyword matching. Combining both (with learned or tuned weights) produces more robust retrieval — a technique used in many production RAG systems.

---

## Tech Stack

- **Orchestration**: LangChain
- **Sparse Retrieval**: BM25 (`rank_bm25`)
- **Dense Retrieval**: ChromaDB + sentence-transformers
- **Fusion**: LangChain `EnsembleRetriever`
- **LLM**: Ollama (recommended) or OpenAI
- **UI**: Streamlit
- **Document Loaders**: pypdf + text

---

## Project Structure

```
insight-rag/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .gitignore
├── .env.example
├── src/
│   ├── config.py           # All configuration in one place
│   ├── document_loader.py  # Load + chunk
│   ├── embeddings.py       # Embedding model
│   ├── vectorstore.py      # Chroma operations
│   ├── retriever.py        # Hybrid (BM25 + Dense) retriever
│   ├── rag_chain.py        # Prompt + generation chain
│   └── evaluator.py        # Lightweight groundedness checks
└── data/
    └── sample/             # Ready-to-use demo documents
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/washington-oyekunle/insight-rag.git
cd insight-rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Choose LLM backend

**Option A — Fully local (Recommended)**

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2               # or mistral / phi3 / gemma2
```

**Option B — OpenAI**

```bash
cp .env.example .env
# Add OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
streamlit run app.py
```

Load the sample documents (or upload your own) and start asking questions.

---

## Sample Questions

With the included company handbook + product overview:

- "What is the remote work policy?"
- "How many days of PTO do employees get?"
- "What parental leave is offered?"
- "Summarize the code of conduct"
- "What are the pricing tiers for Acme Cloud Platform?"
- "How does the cost insights feature work?"

---

## Configuration

All important knobs live in `src/config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | Text splitting |
| `TOP_K` | 4 | Final docs returned |
| `DENSE_K` / `BM25_K` | 6 / 6 | Candidates from each retriever |
| `HYBRID_WEIGHTS` | [0.4, 0.6] | BM25 vs Dense influence |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Local embedding model |
| `OLLAMA_MODEL` | llama3.2 | Local LLM |

Tune the hybrid weights depending on your data: increase BM25 weight for keyword-heavy domains (legal, technical docs), increase dense weight for more semantic / conversational queries.

---

## Evaluation

A lightweight groundedness heuristic is included (`src/evaluator.py`).  
For interview discussions or real projects, mention the natural next steps:

- RAGAS (faithfulness, answer relevancy, context precision/recall)
- Golden question set with human labels
- LLM-as-judge evaluation

---

## Strong Talking Points for Interviews

**Problem**  
Employees and customers waste time searching internal documents. General LLMs hallucinate or give outdated answers.

**Solution**  
RAG grounds generation in actual documents and provides citations. Hybrid retrieval improves recall on both keyword and semantic queries.

**Key Design Decisions**
- Local embeddings → zero cost + privacy
- Persistent Chroma → index survives restarts
- Hybrid (BM25 + dense) → more robust than pure vector search
- Modular code → easy to swap components or add rerankers later

**Natural Extensions** (great to discuss)
1. Cross-encoder / Cohere / BGE reranker on top of hybrid results
2. Metadata filtering (department, date, document type)
3. Agentic / multi-step retrieval
4. Full evaluation suite with RAGAS
5. FastAPI + Docker deployment
6. Observability (LangSmith, Phoenix, or custom tracing)

---

## License

MIT

---

Built as a portfolio project to demonstrate practical AI engineering skills that employers actually care about.
