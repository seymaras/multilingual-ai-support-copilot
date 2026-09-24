from pathlib import Path

from multilingual_support_copilot.pipeline import search_document


def test_search_document_returns_relevant_chunk(tmp_path: Path) -> None:
    file_path = tmp_path / "support.txt"

    file_path.write_text(
        "Customers can contact support by email. "
        "The return period is 14 days. "
        "Our office is located in Berlin.",
        encoding="utf-8",
    )

    results = search_document(
        query="Müşteri desteğine nasıl ulaşabilirim?",
        file_path=file_path,
        chunk_size=6,
        top_k=1,
    )

    assert len(results) == 1
    assert "support" in results[0][0].lower()
    
def test_search_document_accepts_overlap(tmp_path: Path) -> None:
    file_path = tmp_path / "support.txt"

    file_path.write_text(
        "one two three four five six seven eight",
        encoding="utf-8",
    )

    results = search_document(
        query="three four",
        file_path=file_path,
        chunk_size=4,
        overlap=2,
        top_k=2,
    )

    assert len(results) == 2

def test_search_document_uses_reranker_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    file_path = tmp_path / "support.txt"

    file_path.write_text(
        "The return period is 14 days.",
        encoding="utf-8",
    )

    fake_candidates = [
        ("candidate one", 0.8),
        ("candidate two", 0.7),
        ("candidate three", 0.6),
    ]

    fake_reranked = [
        ("candidate two", 2.0),
        ("candidate one", 1.0),
        ("candidate three", 0.0),
    ]

    def fake_retrieve_top_k_chunks(query, chunks, top_k):
        assert top_k == 3
        return fake_candidates

    def fake_rerank_chunks(query, retrieved_chunks):
        assert retrieved_chunks == fake_candidates
        return fake_reranked

    monkeypatch.setattr(
        "multilingual_support_copilot.pipeline.retrieve_top_k_chunks",
        fake_retrieve_top_k_chunks,
    )

    monkeypatch.setattr(
        "multilingual_support_copilot.pipeline.rerank_chunks",
        fake_rerank_chunks,
    )

    results = search_document(
        query="İade süresi nedir?",
        file_path=file_path,
        top_k=2,
        candidate_k=3,
        use_reranker=True,
    )

    assert results == [
        ("candidate two", 2.0),
        ("candidate one", 1.0),
    ]