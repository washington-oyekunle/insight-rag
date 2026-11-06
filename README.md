# InsightRAG 💡

**Production-style Document Q&A with Hybrid Retrieval + Cross-Encoder Reranking**

A high-quality RAG system built as a strong portfolio project for AI / ML / LLM Engineer roles.

Upload documents → Ask natural language questions → Get grounded, streamed answers with source snippets and relevance scores.

---

## Key Features (Tier-1)

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | BM25 (keyword) + Dense vectors fused with Reciprocal Rank Fusion |
| **Cross-Encoder Reranker** | `ms-marco-MiniLM-L-6-v2` reranks candidates for higher precision |
| **Streaming Answers** | Token-by-token response for a modern UX |
| **Rich Sources** | Shows filename, relevance score, and text snippet |
| **Document Management** | See indexed files and chunk counts in the sidebar |
| **Evaluation Suite** | Golden set + retrieval hit rate + groundedness metrics |

---

## Architecture

```
Documents
    ↓
Load + Chunk
    ↓
┌──────────────────┐
│                     │
BM25 Index          Dense Vectors (Chroma)
│                     │
└──────────────────┘
            ↓
  EnsembleRetriever (RRF)
            ↓
  Cross-Encoder Reranker
            ↓
  Top-k chunks + scores
            ↓
  Streaming LLM Generation
            ↓
  Answer + Source Snippets
```

---

## Why This Project Stands Out

Most portfolio RAG projects stop at basic vector search. This one demonstrates techniques used in real production systems:

- Hybrid retrieval (sparse + dense)
- Learned reranking
- Streaming generation
- Evaluation with a golden set
- Clean modular architecture

These are exactly the topics that come up in Applied AI / LLM Engineer interviews.

---

## Quick Start

```bash
git clone https://github.com/washington-oyekunle/insight-rag.git
cd insight-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Recommended: local LLM
ollama pull llama3.2

streamlit run app.py
```

### Run the evaluation suite

```bash
python -m scripts.run_evaluation
```

This runs the system against a golden question set and reports:
- Retrieval hit rate
- Groundedness rate
- Per-question diagnostics

---

## Project Structure

```
insight-rag/
├── app.py                     # Streamlit UI (streaming + sources + doc management)
├── requirements.txt
├── scripts/
│   └── run_evaluation.py     # Golden-set evaluation runner
├── src/
│   ├── config.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py          # Hybrid + Reranker
│   ├── rag_chain.py           # Prompting + streaming
│   └── evaluator.py
└── data/
    ├── sample/                # Demo documents
    └── eval/
        └── golden_set.json    # Evaluation questions
```

---

## Configuration

All important settings live in `src/config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `HYBRID_WEIGHTS` | `[0.4, 0.6]` | BM25 vs Dense influence |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `USE_RERANKER` | `True` | Toggle reranking |
| `TOP_K` | `4` | Final documents after rerank |
| `RERANK_CANDIDATES` | `12` | Candidates fed to reranker |
| `CHUNK_SIZE` | `800` | Chunk size |

---

## Sample Questions

- "What is the remote work policy?"
- "How many days of PTO do employees get?"
- "What parental leave is offered?"
- "What are the pricing tiers for Acme Cloud Platform?"
- "How does the cost insights feature work?"

---

## Interview Talking Points

**Problem**  
Internal knowledge is trapped in documents. Generic LLMs hallucinate or lack access.

**Solution**  
A hybrid RAG pipeline with reranking that returns grounded answers + evidence.

**Key decisions**
- Hybrid retrieval improves recall on both keyword and semantic queries
- Cross-encoder reranking significantly boosts precision of the final context
- Streaming + source snippets create a trustworthy user experience
- Golden-set evaluation shows commitment to measuring quality

**Natural next steps** (great to discuss)
1. Metadata filtering & access control
2. Agentic / multi-step retrieval
3. Full RAGAS evaluation suite
4. FastAPI + Docker deployment
5. Observability (latency, retrieval logs, feedback loops)

---

## License

MIT
