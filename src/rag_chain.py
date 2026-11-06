"""RAG chain construction."""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL, TEMPERATURE, TOP_K


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context.

Rules:
- Use only the information in the context to answer.
- If the context does not contain enough information, say "I don't have enough information in the documents to answer that."
- Be concise but complete.
- When possible, mention which document the information comes from.

Context:
{context}
"""


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Document {i} | Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def get_llm(provider: str = "ollama", openai_api_key: Optional[str] = None):
    """Return an LLM instance.

    provider: "ollama" (default, local) or "openai"
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=TEMPERATURE,
            api_key=openai_api_key,
        )
    else:
        # Default: Ollama (fully local)
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
        )


def create_rag_chain(retriever, llm):
    """Create a simple RAG chain: retrieve → format → prompt → LLM → parse."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def answer_question(chain, question: str) -> str:
    """Run the chain and return the answer."""
    return chain.invoke(question)
