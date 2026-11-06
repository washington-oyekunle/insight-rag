"""Simple evaluation helpers for RAG quality.

These are lightweight checks. For stronger evaluation, integrate RAGAS or a golden test set.
"""

from typing import List
from langchain_core.documents import Document


def check_groundedness(answer: str, context_docs: List[Document], threshold: float = 0.25) -> dict:
    """Heuristic groundedness score based on word overlap.

    Returns a dict with score and whether the answer appears grounded.
    Replace with NLI models or RAGAS for production use.
    """
    if not answer or not context_docs:
        return {"score": 0.0, "is_grounded": False, "note": "Empty answer or context"}

    context_text = " ".join(doc.page_content.lower() for doc in context_docs)
    answer_words = set(w for w in answer.lower().split() if len(w) > 2)
    context_words = set(context_text.split())

    if not answer_words:
        return {"score": 0.0, "is_grounded": False, "note": "Empty answer"}

    overlap = len(answer_words & context_words) / len(answer_words)
    is_grounded = overlap >= threshold

    return {
        "score": round(overlap, 3),
        "is_grounded": is_grounded,
        "note": "Heuristic word overlap — upgrade to RAGAS / NLI for real evaluation",
    }


def print_eval_report(question: str, answer: str, docs: List[Document]) -> None:
    """Pretty-print a small evaluation summary (useful in notebooks)."""
    result = check_groundedness(answer, docs)
    print("\n" + "=" * 55)
    print(f"Question : {question}")
    print(f"Answer   : {answer[:320]}{'...' if len(answer) > 320 else ''}")
    print(f"Sources  : {[d.metadata.get('source') for d in docs]}")
    print(f"Groundedness : {result['score']} (grounded={result['is_grounded']})")
    print("=" * 55 + "\n")
