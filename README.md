# InsightRAG 💡

**Portfolio-grade Document Q&A System**

Hybrid Retrieval · Cross-Encoder Reranking · Query Rewriting · Agentic Routing · Self-Correction · Observability · FastAPI · Docker

Built to demonstrate the skills hiring managers look for in Applied AI / LLM Engineer roles.

---

## Feature Matrix

| Tier | Capabilities |
|------|----------------|
| **1** | Hybrid BM25+Dense, Cross-encoder rerank, Streaming, Source snippets, Doc management, Golden-set eval |
| **2** | Query rewriting, Metadata filtering, FastAPI, Docker Compose, Chat persistence |
| **3** | Agentic routing, Self-correcting retrieval, Structured query logs, `/stats` observability |

---

## Architecture

```
User Message
     ↓
Agentic Router ─► CHITCHAT / CLARIFY / RETRIEVE
     ↓ (RETRIEVE)
Query Rewriter
     ↓
Hybrid Retriever (BM25 + Dense / RRF)
     ↓
Metadata Filter (optional)
     ↓
Cross-Encoder Reranker
     ↓
LLM Answer
     ↓
Groundedness Check ─► weak? → Reformulate + Retrieve again (self-correct)
     ↓
Observability Log (latency, sources, scores, corrections)
```

---

## Quick Start

### Local
```bash
git clone https://github.com/washington-oyekunle/insight-rag.git
cd insight-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.2

streamlit run app.py
# API:
uvicorn api.main:app --reload --port 8000
```

### Docker
```bash
docker compose up --build
```
- UI: http://localhost:8501
- API: http://localhost:8000/docs

---

## API Highlights

| Endpoint | Description |
|----------|-------------|
| `POST /query` | Full agentic + self-correcting Q&A |
| `POST /query/stream` | Streaming responses |
| `GET /stats` | Avg latency, correction rate, groundedness |
| `GET /logs` | Recent structured query events |
| `POST /index/sample` | Index demo documents |
| `POST /index/upload` | Upload & index files |

---

## Evaluation

```bash
python -m scripts.run_evaluation
```

---

## Project Structure

```
insight-rag/
├── api/main.py
├── app.py
├── Dockerfile
├── docker-compose.yml
├── scripts/run_evaluation.py
├── src/
│   ├── pipeline.py         # Orchestrates full Tier-3 flow
│   ├── agent.py            # Routing + reformulation
│   ├── observability.py    # JSONL logs + stats
│   ├── retriever.py
│   ├── query_rewrite.py
│   ├── rag_chain.py
│   ├── chat_history.py
│   └── evaluator.py
└── data/
    ├── sample/
    └── eval/golden_set.json
```

---

## Interview Talking Points

1. **Hybrid retrieval + reranking** — production pattern for high recall *and* precision
2. **Query rewriting** — bridges conversational questions and search terms
3. **Agentic routing** — not every message should hit the vector DB
4. **Self-correction** — detect weak groundedness and retry with a reformulated query
5. **Observability** — log latency, sources, corrections; expose `/stats`
6. **API + Docker** — ship beyond notebooks

---

## License

MIT
