# InsightRAG 💡

**Production-style Document Q&A**  
Hybrid Retrieval · Cross-Encoder Reranking · Query Rewriting · FastAPI · Docker

A portfolio-grade RAG system designed to demonstrate real Applied AI / LLM engineering skills.

---

## Features

### Tier-1
- Hybrid search (BM25 + Dense vectors)
- Cross-encoder reranking
- Streaming answers
- Rich source snippets + relevance scores
- Document management UI
- Golden-set evaluation suite

### Tier-2
- **Query rewriting** — improves retrieval on vague questions
- **Metadata filtering** — restrict search to selected documents
- **FastAPI backend** — clean REST API with streaming
- **Docker + docker-compose** — one-command startup
- **Chat history persistence** — sessions saved to disk

---

## Architecture

```
User Question
      ↓
Query Rewriter (LLM)
      ↓
Hybrid Retriever (BM25 + Dense / RRF)
      ↓
Metadata Filter (optional)
      ↓
Cross-Encoder Reranker
      ↓
Streaming LLM Generation
      ↓
Answer + Source Snippets + Scores
```

---

## Quick Start (Local)

```bash
git clone https://github.com/washington-oyekunle/insight-rag.git
cd insight-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Local LLM (recommended)
ollama pull llama3.2

# UI
streamlit run app.py

# API (separate terminal)
uvicorn api.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## Quick Start (Docker)

```bash
# Ensure Ollama is running on the host
docker compose up --build
```

- UI:  http://localhost:8501  
- API: http://localhost:8000  
- Docs: http://localhost:8000/docs

---

## API Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health + index status |
| GET | `/documents` | Listed indexed docs |
| POST | `/index/sample` | Index bundled sample docs |
| POST | `/index/upload` | Upload & index files |
| POST | `/query` | Non-streaming Q&A |
| POST | `/query/stream` | Streaming Q&A |
| DELETE | `/documents` | Clear index |
| POST | `/sessions` | Create chat session |
| GET | `/sessions/{id}` | Load chat history |

Example:
```bash
# Index samples
curl -X POST http://localhost:8000/index/sample

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the remote work policy?"}'
```

---

## Evaluation

```bash
python -m scripts.run_evaluation
```

Reports retrieval hit rate and groundedness against `data/eval/golden_set.json`.

---

## Project Structure

```
insight-rag/
├── api/main.py              # FastAPI backend
├── app.py                   # Streamlit UI
├── Dockerfile
├── docker-compose.yml
├── scripts/run_evaluation.py
├── src/
│   ├── pipeline.py          # Shared RAG pipeline
│   ├── retriever.py          # Hybrid + rerank + filter
│   ├── query_rewrite.py      # Query rewriting
│   ├── rag_chain.py
│   ├── chat_history.py       # Session persistence
│   ├── evaluator.py
│   └── ...
└── data/
    ├── sample/
    └── eval/golden_set.json
```

---

## Interview Talking Points

- **Hybrid retrieval** improves recall on keyword *and* semantic queries
- **Cross-encoder reranking** boosts precision of the final context window
- **Query rewriting** helps with conversational / vague user input
- **FastAPI + Docker** shows you can ship beyond notebooks
- **Evaluation harness** proves you measure quality, not just demos
- **Metadata filtering** is a step toward access control and multi-tenant search

---

## License

MIT
