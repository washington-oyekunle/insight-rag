"""RAG chain construction with streaming support."""

from typing import Optional, Iterator, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL, TEMPERATURE


SYSTEM_PROMPT = """You are a precise and helpful assistant that answers questions using only the provided context.

Guidelines:
- Base your answer strictly on the context below. Do not use outside knowledge.
- If the context is insufficient, reply: "I don't have enough information in the documents to answer that."
- Prefer concise, well-structured answers.
- When relevant, mention the source document name.
- If multiple documents contain related information, synthesize them clearly.

Context:
{context}
"""


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string with sources."""
    if not docs:
        return "No relevant context found."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Document {i} | Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def get_llm(provider: str = "ollama", openai_api_key: Optional[str] = None):
    """Return an LLM instance that supports streaming."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=TEMPERATURE,
            api_key=openai_api_key,
            streaming=True,
        )
    else:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
        )


def create_rag_chain(llm):
    """Create a RAG chain that expects context + question.

    Note: retrieval is done outside the chain so we can show sources
    and support streaming cleanly.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def stream_answer(chain, question: str, context_docs: List[Document]) -> Iterator[str]:
    """Stream the answer token by token."""
    context = format_docs(context_docs)
    for chunk in chain.stream({"question": question, "context": context}):
        yield chunk


def answer_question(chain, question: str, context_docs: List[Document]) -> str:
    """Non-streaming full answer (useful for evaluation)."""
    context = format_docs(context_docs)
    return chain.invoke({"question": question, "context": context})
