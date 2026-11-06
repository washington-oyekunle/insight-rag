"""Run golden-set evaluation for InsightRAG.

Usage:
    python -m scripts.run_evaluation

Requires that sample documents have been processed (or will process them automatically).
"""

from pathlib import Path
import sys

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SAMPLE_DIR
from src.document_loader import load_and_chunk
from src.vectorstore import get_vectorstore, add_documents
from src.retriever import create_hybrid_retriever, retrieve_and_rerank
from src.rag_chain import get_llm, create_rag_chain, answer_question
from src.evaluator import run_golden_evaluation, load_golden_set


def main():
    print("Loading sample documents...")
    files = list(SAMPLE_DIR.glob("*"))
    chunks = load_and_chunk(files)
    print(f"  {len(chunks)} chunks loaded")

    print("Building vector store + hybrid retriever...")
    vectorstore = get_vectorstore()
    add_documents(vectorstore, chunks)
    hybrid = create_hybrid_retriever(documents=chunks, vectorstore=vectorstore)

    print("Loading LLM...")
    llm = get_llm(provider="ollama")
    chain = create_rag_chain(llm)

    def retrieve_fn(query: str):
        ranked = retrieve_and_rerank(hybrid, query)
        return [doc for doc, score in ranked]

    def answer_fn(question: str, docs):
        return answer_question(chain, question, docs)

    print("\nRunning golden-set evaluation...\n")
    summary = run_golden_evaluation(
        retriever=hybrid,
        chain=chain,
        retrieve_fn=retrieve_fn,
        answer_fn=answer_fn,
    )

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Questions              : {summary['num_questions']}")
    print(f"Retrieval hit rate     : {summary['retrieval_hit_rate']:.1%}")
    print(f"Groundedness rate      : {summary['groundedness_rate']:.1%}")
    print()

    for i, r in enumerate(summary["results"], 1):
        print(f"{i}. {r['question']}")
        print(f"   Retrieval hit : {r['retrieval']['hit']} | sources: {r['retrieval']['retrieved_sources']}")
        print(f"   Groundedness  : {r['groundedness']['score']}")
        print(f"   Phrase hits   : {r['phrase_hits']}")
        print(f"   Answer        : {r['answer'][:180]}{'...' if len(r['answer']) > 180 else ''}")
        print()


if __name__ == "__main__":
    main()
