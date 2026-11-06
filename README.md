# InsightRAG 💡

**Intelligent Document Q&A System using Retrieval-Augmented Generation (RAG)**

A clean, modular, production-style RAG application built for portfolio and job applications.  
Upload your documents → Ask natural language questions → Get grounded answers with source citations.

> Designed to demonstrate the skills hiring managers look for in Applied AI / ML Engineer / LLM Engineer roles:  
> retrieval systems, embeddings, vector databases, prompt engineering, evaluation, clean architecture, and a working UI.

---

## 💻 Demo

```bash
# After setup
streamlit run app.py
```

Upload PDFs or text files, then chat with your documents.

---

## 🎯 Why This Project Matters for Jobs

Most junior/mid-level candidates show notebooks.  
This project shows:

| Skill | How it's demonstrated |
|-------|-----------------------|
| **RAG Architecture** | Full pipeline: load → chunk → embed → retrieve → generate |
| **Vector Search** | ChromaDB + sentence-transformers embeddings |
| **LLM Integration** | Support for Ollama (local/free) + OpenAI |
| **Evaluation Mindset** | Simple faithfulness / relevance checks |
| **Software Engineering** | Modular code, config, clean separation of concerns |
| **UI/Product Sense** | Streamlit chat interface with sources |
| **Reproducibility** | Clear setup, requirements, sample data |

---

## 📦 Tech Stack

- **Orchestration**: LangChain
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **Vector Store**: ChromaDB (persistent)
- **LLM**: Ollama (recommended, fully local) or OpenAI
- **UI**: Streamlit
- **Document Loaders**: PDF + TXT

---

## 📁 Project Structure

```
insight-rag/
├── app.py                 # Streamlit UI
├── requirements.txt
├── .gitignore
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration
│   ├── document_loader.py # Load & chunk documents
│   ├── embeddings.py      # Embedding model
│   ├── vectorstore.py     # ChromaDB operations
│   ├── rag_chain.py       # RAG pipeline
│   └── evaluator.py       # Simple evaluation helpers
├── data/
│   └── sample/            # Sample documents for demo
└── notebooks/             # Optional exploration
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/washington-oyekunle/insight-rag.git
cd insight-rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Choose your LLM backend

**Option A — Fully local & free (Recommended)**

1. Install [Ollama](https://ollama.com/)
2. Pull a model:
   ```bash
   ollama pull llama3.2        # or mistral, phi3, etc.
   ```
3. Make sure Ollama is running (`ollama serve`)

**Option B — OpenAI**

```bash
cp .env.example .env
# Add your OPENAI_API_KEY
```

### 3. Run the app

```bash
streamlit run app.py
```

Upload the sample documents in `data/sample/` or your own PDFs/TXTs and start asking questions.

---

## 🔍 Architecture Overview

```
Documents (PDF/TXT)
        ↓
   Load + Chunk (RecursiveCharacterTextSplitter)
        ↓
   Embed (all-MiniLM-L6-v2)
        ↓
   Store in ChromaDB (persistent)
        ↓
User Question → Retrieve top-k relevant chunks
        ↓
   Prompt + Context + Question → LLM
        ↓
   Answer + Source Citations
```

---

## 📊 Evaluation

A lightweight evaluation helper is included in `src/evaluator.py`.

It checks:
- Whether the answer is grounded in the retrieved context (faithfulness)
- Basic relevance signals

You can extend this with RAGAS, DeepEval, or custom golden datasets for production-grade evaluation.

---

## 🛠️ Configuration

Key settings live in `src/config.py`:

- Chunk size / overlap
- Number of retrieved documents (`k`)
- Embedding model name
- LLM model name (for Ollama)
- Persistence directory for Chroma

---

## 📝 Sample Questions (with included data)

After loading the sample company handbook:

- "What is the remote work policy?"
- "How many days of paid time off do employees get?"
- "What is the process for requesting parental leave?"
- "Summarize the code of conduct."

---

## 🚀 Possible Improvements (Great for Interviews)

Mention these when discussing the project:

1. **Hybrid search** (BM25 + dense vectors)
2. **Reranking** (cross-encoder or Cohere/BGE reranker)
3. **Advanced evaluation** (RAGAS: faithfulness, answer relevancy, context precision)
4. **Metadata filtering** (by document type, date, department)
5. **Agentic RAG** / multi-step retrieval
6. **Streaming responses**
7. **Docker + FastAPI** backend for production
8. **Observability** (LangSmith / Phoenix / custom logging)

---

## 💡 How to Talk About This in Interviews

**Problem**: Companies have large amounts of unstructured documents. Employees waste time searching or get outdated answers from general LLMs.

**Solution**: RAG grounds the LLM in the actual documents, provides citations, and reduces hallucinations.

**Your contributions / decisions**:
- Chose local embeddings for zero cost and privacy
- Used persistent Chroma so the index survives restarts
- Separated concerns so the system is easy to extend
- Added evaluation hooks to measure quality

---

## License

MIT

---

Built as a portfolio project to demonstrate practical AI engineering skills.
