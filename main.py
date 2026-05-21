"""
main.py - CLI benchmark bez Gradio interfejsa.
Korisno za testiranje u terminalu.
"""

import logging
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from src.config import BOOK_PATH, NUM_TEST_QUESTIONS
from src.data_preparer import load_and_split_pdf, generate_test_questions
from src.tool_factory import build_vector_store, build_bm25_retriever, create_tools
from src.agent_orchestrator import build_agent, run_agent
from src.evaluator import run_ragas_evaluation, compute_summary


def main():
    print("=" * 60)
    print("  Agentic RAG Benchmark - Diplomski rad")
    print("=" * 60)

    # 1. Učitaj i podeli PDF
    if not Path(BOOK_PATH).exists():
        print(f"[GREŠKA] PDF nije pronađen na: {BOOK_PATH}")
        print("Postavi svoju knjigu na data/book.pdf i ponovo pokreni.")
        return

    docs = load_and_split_pdf(BOOK_PATH)

    # 2. Izgradi indekse
    vector_store   = build_vector_store(docs)
    bm25_retriever = build_bm25_retriever(docs)
    tools          = create_tools(vector_store, bm25_retriever)
    agent          = build_agent(tools)

    # 3. Generiši test pitanja
    print(f"\nGenerišem {NUM_TEST_QUESTIONS} test pitanja...")
    test_set = generate_test_questions(docs, n=NUM_TEST_QUESTIONS)

    # 4. Pokreni agenta na svakom pitanju
    print("\nPokrećem agenta na test pitanjima...")
    results = []
    for i, item in enumerate(test_set, 1):
        print(f"\n[{i}/{len(test_set)}] Pitanje: {item['question']}")
        agent_result = run_agent(agent, item["question"])
        print(f"  Odgovor: {agent_result['answer'][:200]}...")
        results.append({
            "question":     item["question"],
            "answer":       agent_result["answer"],
            "contexts":     agent_result["contexts"],
            "ground_truth": item["ground_truth"],
        })

    # 5. RAGAS evaluacija
    print("\nPokrećem RAGAS evaluaciju...")
    df = run_ragas_evaluation(results)
    summary = compute_summary(df)

    print("\n" + "=" * 60)
    print("  RAGAS Rezultati")
    print("=" * 60)
    for metric, score in summary.items():
        print(f"  {metric:<30} {score:.4f}")

    # 6. Sačuvaj rezultate
    output_path = "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
    print(f"\nRezultati sačuvani u: {output_path}")


if __name__ == "__main__":
    main()
