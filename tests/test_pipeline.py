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