"""Configuration for InsightRAG."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
EVAL_DIR = DATA_DIR / "eval"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 4                    # final docs after reranking
DENSE_K = 8                  # candidates from vector search
BM25_K = 8                   # candidates from BM25
RERANK_CANDIDATES = 12       # max docs sent to the reranker

# Hybrid search weights [bm25_weight, dense_weight]
HYBRID_WEIGHTS = [0.4, 0.6]

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reranker (Cross-Encoder)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = True

# LLM settings
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.2
