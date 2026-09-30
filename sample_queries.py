"""
sample_queries.py - Run 6 sample queries and print structured output.

Used to generate the sample_outputs.json for the README.

Run with:
    python sample_queries.py
"""

import json
import time
from rag_pipeline import run_rag_pipeline

SAMPLE_QUESTIONS = [
    "What is Agentic AI?",
    "How does Agentic AI differ from traditional AI?",
    "What are the key components of an AI agent?",
    "What are the use cases of Agentic AI?",
    "What challenges or risks are mentioned in the eBook?",
    "Who is the prime minister of India?",   # testing out-of-scope grounding
]


def main():
    results = []
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"Q{i}: {question}")
        print("="*60)
        t0 = time.perf_counter()
        result = run_rag_pipeline(question)
        latency = round((time.perf_counter() - t0) * 1000, 1)

        print(f"Answer     : {result['answer']}")
        print(f"Confidence : {result['confidence']:.4f}")
        print(f"Latency    : {latency} ms")
        print(f"Top chunk  : {result['retrieved_chunks'][0]['text'][:120]}...")

        results.append({
            "question": question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "latency_ms": latency,
            "retrieved_chunks": result["retrieved_chunks"],
        })

    # Persist to JSON for the README
    with open("sample_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n\n✅ Results saved to sample_outputs.json")


if __name__ == "__main__":
    main()
