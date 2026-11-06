"""Simple file-based chat history persistence."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4

from src.config import CHAT_HISTORY_DIR


def _history_path(session_id: str) -> Path:
    return CHAT_HISTORY_DIR / f"{session_id}.json"


def create_session() -> str:
    """Create a new chat session and return its ID."""
    session_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    path = _history_path(session_id)
    data = {"session_id": session_id, "created_at": datetime.utcnow().isoformat() + "Z", "messages": []}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return session_id


def load_history(session_id: str) -> List[Dict[str, Any]]:
    path = _history_path(session_id)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("messages", [])


def append_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict]] = None,
) -> None:
    path = _history_path(session_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"session_id": session_id, "created_at": datetime.utcnow().isoformat() + "Z", "messages": []}

    msg = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if sources:
        msg["sources"] = sources

    data["messages"].append(msg)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_sessions() -> List[Dict[str, str]]:
    sessions = []
    for path in sorted(CHAT_HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions.append(
                {
                    "session_id": data.get("session_id", path.stem),
                    "created_at": data.get("created_at", ""),
                    "message_count": len(data.get("messages", [])),
                }
            )
        except Exception:
            continue
    return sessions
