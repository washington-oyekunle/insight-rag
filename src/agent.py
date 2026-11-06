"""Lightweight agentic controls for RAG.

- Route: decide whether retrieval is needed vs clarification
- Self-correct: if answer is weakly grounded, reformulate query and retry once
"""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import GROUNDEDNESS_THRESHOLD


ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a routing agent for a document Q&A system.\n"
            "Classify the user message into exactly one label:\n"
            "- RETRIEVE: a question that should be answered from documents\n"
            "- CLARIFY: too vague / missing key details; ask a clarifying question\n"
            "- CHITCHAT: greeting or small talk unrelated to documents\n"
            "Reply with only the label.",
        ),
        ("human", "{question}"),
    ]
)


REFORMULATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The previous search failed to find good evidence. "
            "Rewrite the question into a different, more specific search query "
            "using alternative keywords or phrasing. Output ONLY the new query.",
        ),
        (
            "human",
            "Original question: {question}\nPrevious search query: {search_query}\nNew search query:",
        ),
    ]
)


def route_question(question: str, llm) -> str:
    """Return RETRIEVE | CLARIFY | CHITCHAT."""
    try:
        chain = ROUTE_PROMPT | llm | StrOutputParser()
        label = chain.invoke({"question": question}).strip().upper()
        if "CLARIFY" in label:
            return "CLARIFY"
        if "CHITCHAT" in label:
            return "CHITCHAT"
        return "RETRIEVE"
    except Exception:
        return "RETRIEVE"


def reformulate_query(question: str, search_query: str, llm) -> str:
    """Produce an alternative search query after a weak retrieval/answer."""
    try:
        chain = REFORMULATE_PROMPT | llm | StrOutputParser()
        new_q = chain.invoke(
            {"question": question, "search_query": search_query}
        ).strip()
        if not new_q or len(new_q) > 500:
            return search_query
        return new_q
    except Exception:
        return search_query


def needs_correction(groundedness: dict, threshold: float = GROUNDEDNESS_THRESHOLD) -> bool:
    if not groundedness:
        return True
    return not groundedness.get("is_grounded", False) or groundedness.get("score", 0) < threshold
