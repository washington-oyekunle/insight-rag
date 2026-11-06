"""Query rewriting to improve retrieval.

Rewrites the user question into a clearer search-oriented query.
This helps with vague, conversational, or underspecified questions.
"""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import USE_QUERY_REWRITE


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a search query optimizer. Rewrite the user's question into a clear, "
            "specific search query that will retrieve the most relevant document passages. "
            "Keep key entities, terms, and intent. Output ONLY the rewritten query, nothing else.",
        ),
        ("human", "{question}"),
    ]
)


def rewrite_query(question: str, llm, enabled: bool = USE_QUERY_REWRITE) -> str:
    """Rewrite a user question into a better retrieval query.

    Falls back to the original question on any failure.
    """
    if not enabled or not question.strip():
        return question

    try:
        chain = REWRITE_PROMPT | llm | StrOutputParser()
        rewritten = chain.invoke({"question": question}).strip()
        # Safety: if model returns something empty or huge, keep original
        if not rewritten or len(rewritten) > 500:
            return question
        return rewritten
    except Exception:
        return question
