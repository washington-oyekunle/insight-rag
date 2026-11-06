"""Configuration for InsightRAG."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 4                    # final number of docs returned by hybrid retriever
DENSE_K = 6                  # candidates from vector search
BM25_K = 6                   # candidates from BM25

# Hybrid search weights (must sum roughly to 1.0)
# Higher BM25 weight helps with keyword / exact-term queries
# Higher dense weight helps with semantic / paraphrased queries
HYBRID_WEIGHTS = [0.4, 0.6]  # [bm25_weight, dense_weight]

# Embeddings (local, free, good quality for the size)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM settings
# For Ollama (recommended)
OLLAMA_MODEL = "llama3.2"          # change to mistral, phi3, gemma2, etc.
OLLAMA_BASE_URL = "http://localhost:11434"

# Temperature for generation
TEMPERATURE = 0.2
