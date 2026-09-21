from multilingual_support_copilot.retriever import retrieve_top_chunk


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