"""InsightRAG FastAPI backend (Tier-3).

Adds:
- Agentic routing + self-correction (via pipeline)
- Observability endpoints (/logs, /stats)
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import UPLOAD_DIR, API_HOST, API_PORT
from src.pipeline import pipeline
from src.chat_history import create_session, load_history, append_message, list_sessions
from src.observability import read_logs, compute_stats

app = FastAPI(
    title="InsightRAG API",
    description="Hybrid RAG with reranking, query rewriting, self-correction, and observability",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    provider: str = "ollama"
    openai_api_key: Optional[str] = None
    source_filter: Optional[List[str]] = None
    use_rewrite: bool = True
    use_self_correction: bool = True
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    search_query: str
    sources: List[dict]
    groundedness: dict
    route: str = "RETRIEVE"
    self_corrected: bool = False
    correction_rounds: int = 0
    session_id: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "ready": pipeline.is_ready,
        "documents": pipeline.doc_stats,
        "chunks": len(pipeline.chunks),
    }


@app.get("/documents")
def get_documents():
    return {
        "ready": pipeline.is_ready,
        "doc_stats": pipeline.doc_stats,
        "total_chunks": len(pipeline.chunks),
    }


@app.post("/index/sample")
def index_sample():
    result = pipeline.index_sample()
    if result["chunks"] == 0:
        raise HTTPException(status_code=400, detail="No sample documents found")
    return {"message": "Sample documents indexed", **result}


@app.post("/index/upload")
async def index_upload(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in {".pdf", ".txt", ".md"}:
            continue
        dest = UPLOAD_DIR / (f.filename or "upload.txt")
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)

    if not saved:
        raise HTTPException(status_code=400, detail="No valid PDF/TXT/MD files")

    result = pipeline.index_files(saved)
    return {"message": "Documents indexed", **result}


@app.delete("/documents")
def clear_documents():
    pipeline.clear()
    return {"message": "Index cleared"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        result = pipeline.query(
            question=req.question,
            provider=req.provider,
            openai_api_key=req.openai_api_key or os.getenv("OPENAI_API_KEY"),
            source_filter=req.source_filter,
            use_rewrite=req.use_rewrite,
            use_self_correction=req.use_self_correction,
            stream=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    session_id = req.session_id
    if session_id and result.get("answer"):
        append_message(session_id, "user", req.question)
        append_message(session_id, "assistant", result["answer"], sources=result.get("sources"))

    return QueryResponse(
        answer=result.get("answer", ""),
        search_query=result.get("search_query", ""),
        sources=result.get("sources", []),
        groundedness=result.get("groundedness", {}),
        route=result.get("route", "RETRIEVE"),
        self_corrected=result.get("self_corrected", False),
        correction_rounds=result.get("correction_rounds", 0),
        session_id=session_id,
    )


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    try:
        result = pipeline.query(
            question=req.question,
            provider=req.provider,
            openai_api_key=req.openai_api_key or os.getenv("OPENAI_API_KEY"),
            source_filter=req.source_filter,
            use_rewrite=req.use_rewrite,
            use_self_correction=False,  # full correction needs non-stream path
            stream=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    def event_generator():
        full = []
        for chunk in result["stream"]:
            full.append(chunk)
            yield chunk
        if req.session_id:
            append_message(req.session_id, "user", req.question)
            append_message(
                req.session_id,
                "assistant",
                "".join(full),
                sources=result.get("sources"),
            )

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.post("/sessions")
def new_session():
    return {"session_id": create_session()}


@app.get("/sessions")
def get_sessions():
    return {"sessions": list_sessions()}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    return {"session_id": session_id, "messages": load_history(session_id)}


@app.get("/logs")
def get_logs(limit: int = 50):
    return {"logs": read_logs(limit=limit)}


@app.get("/stats")
def get_stats():
    return compute_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
