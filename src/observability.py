"""Observability: structured query logging and simple stats."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.config import LOG_DIR, ENABLE_LOGGING

LOG_FILE = LOG_DIR / "queries.jsonl"


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def log_query(event: Dict[str, Any]) -> str:
    """Append a structured query event to the JSONL log.

    Returns the event_id.
    """
    if not ENABLE_LOGGING:
        return event.get("event_id", "")

    event_id = event.get("event_id") or uuid4().hex[:12]
    record = {
        "event_id": event_id,
        "timestamp": _now(),
        **event,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return event_id


def read_logs(limit: int = 50) -> List[Dict[str, Any]]:
    if not LOG_FILE.exists():
        return []

    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    events = []
    for line in lines[-limit:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(events))


def compute_stats() -> Dict[str, Any]:
    events = read_logs(limit=1000)
    if not events:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0,
            "self_correction_rate": 0,
            "avg_groundedness": 0,
        }

    latencies = [e["latency_ms"] for e in events if "latency_ms" in e]
    corrections = [e for e in events if e.get("self_corrected")]
    grounded = [
        e["groundedness"]["score"]
        for e in events
        if isinstance(e.get("groundedness"), dict) and "score" in e["groundedness"]
    ]

    return {
        "total_queries": len(events),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "self_correction_rate": round(len(corrections) / len(events), 3),
        "avg_groundedness": round(sum(grounded) / len(grounded), 3) if grounded else 0,
        "recent": events[:10],
    }


class Timer:
    """Simple context-manager style timer."""

    def __init__(self):
        self.start = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self.start) * 1000, 1)
