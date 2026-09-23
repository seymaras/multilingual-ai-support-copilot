import json
from pathlib import Path

from multilingual_support_copilot.retriever import retrieve_top_k_chunks


def evaluate_retrieval(
    document_path: Path,
    queries_path: Path,
    top_k: int = 3,
) -> float:
    document_text = document_path.read_text(encoding="utf-8")

    chunks = [
        line.strip()
        for line in document_text.splitlines()
        if line.strip()
    ]

    queries = json.loads(
        queries_path.read_text(encoding="utf-8")
    )

    successful_queries = 0

    for item in queries:
        results = retrieve_top_k_chunks(
            query=item["query"],
            chunks=chunks,
            top_k=top_k,
        )

        retrieved_texts = [
            chunk
            for chunk, _ in results
        ]

        if item["expected_evidence"] in retrieved_texts:
            successful_queries += 1

    return successful_queries / len(queries)