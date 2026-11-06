"""Simple evaluation helpers for RAG quality.

These are lightweight checks you can expand with RAGAS or custom golden sets.
"""

from typing import List
from langchain_core.documents import Document


def check_groundedness(answer: str, context_docs: List[Document], threshold: float = 0.3) -> dict:
    """Very simple heuristic: measure overlap between answer and context.

    Returns a dict with score and whether it looks grounded.
    This is intentionally basic — replace with proper NLI / RAGAS in production.
    """
    if not answer or not context_docs:
        return {"score": 0.0, "is_grounded": False, "note": "Empty answer or context"}

    context_text = " ".join(doc.page_content.lower() for doc in context_docs)
    answer_words = set(answer.lower().split())
    context_words = set(context_text.split())

    if not answer_words:
        return {"score": 0.0, "is_grounded": False, "note": "Empty answer"}

    overlap = len(answer_words & context_words) / len(answer_words)
    is_grounded = overlap >= threshold

    return {
        "score": round(overlap, 3),
        "is_grounded": is_grounded,
        "note": "Heuristic word overlap — replace with stronger metrics for real eval",
    }


def print_eval_report(question: str, answer: str, docs: List[Document]) -> None:
    """Pretty-print a small evaluation summary."""
    result = check_groundedness(answer, docs)
    print("\n" + "=" * 50)
    print(f"Question : {question}")
    print(f"Answer   : {answer[:300]}{'...' if len(answer) > 300 else ''}")
    print(f"Sources  : {[d.metadata.get('source') for d in docs]}")
    print(f"Groundedness score : {result['score']} (grounded={result['is_grounded']})")
    print("=" * 50 + "\n")
