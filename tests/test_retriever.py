from multilingual_support_copilot.retriever import (
    retrieve_top_chunk,
    retrieve_top_k_chunks,
)


def test_retrieve_top_chunk_returns_most_relevant_chunk() -> None:
    chunks = [
        "Customers can reset their password by email.",
        "The return period is 14 days.",
        "Our office is located in Berlin.",
    ]

    query = "İade için kaç günüm var?"

    best_chunk, score = retrieve_top_chunk(query, chunks)

    assert best_chunk == "The return period is 14 days."
    assert isinstance(score, float)

def test_retrieve_top_k_chunks_returns_ranked_results() -> None:
    chunks = [
        "Customers can reset their password by email.",
        "The return period is 14 days.",
        "Returned products must be unused.",
        "Our office is located in Berlin.",
    ]

    query = "İade hakkında bilgi istiyorum."

    results = retrieve_top_k_chunks(
        query,
        chunks,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][1] >= results[1][1]