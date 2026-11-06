"""Evaluation utilities for InsightRAG.

Includes:
- Lightweight groundedness heuristic
- Golden-set evaluation runner
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from src.config import EVAL_DIR


def check_groundedness(
    answer: str,
    context_docs: List[Document],
    threshold: float = 0.25,
) -> dict:
    """Heuristic groundedness score based on word overlap."""
    if not answer or not context_docs:
        return {"score": 0.0, "is_grounded": False, "note": "Empty answer or context"}

    context_text = " ".join(doc.page_content.lower() for doc in context_docs)
    answer_words = set(w for w in answer.lower().split() if len(w) > 2)
    context_words = set(context_text.split())

    if not answer_words:
        return {"score": 0.0, "is_grounded": False, "note": "Empty answer"}

    overlap = len(answer_words & context_words) / len(answer_words)
    return {
        "score": round(overlap, 3),
        "is_grounded": overlap >= threshold,
        "note": "Heuristic word overlap",
    }


def load_golden_set(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the golden evaluation set."""
    path = path or (EVAL_DIR / "golden_set.json")
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(
    retrieved_docs: List[Document],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """Simple retrieval metric: did we retrieve at least one expected source?"""
    retrieved_sources = {d.metadata.get("source", "") for d in retrieved_docs}
    hits = [s for s in expected_sources if s in retrieved_sources]
    return {
        "hit": len(hits) > 0,
        "retrieved_sources": list(retrieved_sources),
        "expected_sources": expected_sources,
        "matched": hits,
    }


def run_golden_evaluation(
    retriever,
    chain,
    retrieve_fn,
    answer_fn,
    golden_set: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Run evaluation against the golden set.

    retrieve_fn(query) -> List[Document]
    answer_fn(question, docs) -> str
    """
    if golden_set is None:
        golden_set = load_golden_set()

    if not golden_set:
        return {"error": "No golden set found", "results": []}

    results = []
    retrieval_hits = 0
    grounded_count = 0

    for item in golden_set:
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        expected_answer_contains = item.get("expected_answer_contains", [])

        docs = retrieve_fn(question)
        answer = answer_fn(question, docs)

        retrieval = evaluate_retrieval(docs, expected_sources)
        grounded = check_groundedness(answer, docs)

        # Simple answer quality: does the answer contain key phrases?
        answer_lower = answer.lower()
        phrase_hits = [
            p for p in expected_answer_contains if p.lower() in answer_lower
        ]

        if retrieval["hit"]:
            retrieval_hits += 1
        if grounded["is_grounded"]:
            grounded_count += 1

        results.append(
            {
                "question": question,
                "answer": answer,
                "retrieval": retrieval,
                "groundedness": grounded,
                "phrase_hits": phrase_hits,
                "phrase_hit_rate": len(phrase_hits) / max(len(expected_answer_contains), 1),
            }
        )

    n = len(golden_set)
    summary = {
        "num_questions": n,
        "retrieval_hit_rate": round(retrieval_hits / n, 3) if n else 0,
        "groundedness_rate": round(grounded_count / n, 3) if n else 0,
        "results": results,
    }
    return summary
