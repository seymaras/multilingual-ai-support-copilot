from pathlib import Path
from typing import Any

import pytest

from multilingual_support_copilot import pipeline
from multilingual_support_copilot.config import Settings
from multilingual_support_copilot.errors import (
    DocumentNotFoundError,
    EmptyDocumentError,
    InvalidDocumentIdError,
)
from multilingual_support_copilot.pipeline import search_chunks, search_document


def _write(directory: Path, document_id: str, text: str) -> None:
    (directory / f"{document_id}.txt").write_text(text, encoding="utf-8")


def test_search_document_returns_relevant_chunk(documents_dir: Path, fake_embedder: Any) -> None:
    _write(
        documents_dir,
        "support",
        "Customers can contact support by email. "
        "The return period is 14 days. "
        "Our office is located in Berlin.",
    )

    results = search_document(
        query="How do I contact support?",
        document_id="support",
        top_k=1,
        settings=Settings(chunk_size=6),
    )

    assert len(results) == 1
    assert "support" in results[0][0].lower()


def test_search_document_uses_chunk_settings(
    documents_dir: Path, fake_embedder: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(documents_dir, "numbers", "one two three four five six seven eight")
    monkeypatch.setenv("COPILOT_CHUNK_SIZE", "4")
    monkeypatch.setenv("COPILOT_CHUNK_OVERLAP", "2")

    results = search_document(query="three four", document_id="numbers", top_k=5)

    assert sorted(chunk for chunk, _ in results) == [
        "five six seven eight",
        "one two three four",
        "three four five six",
    ]


def test_search_document_defaults_top_k_from_settings(
    documents_dir: Path, fake_embedder: Any
) -> None:
    _write(documents_dir, "doc", " ".join(f"word{i}" for i in range(20)))

    results = search_document(
        query="word1", document_id="doc", settings=Settings(chunk_size=2, default_top_k=4)
    )

    assert len(results) == 4


def test_search_document_rejects_unknown_document(documents_dir: Path) -> None:
    with pytest.raises(DocumentNotFoundError):
        search_document(query="anything", document_id="missing")


def test_search_document_rejects_path_traversal(documents_dir: Path) -> None:
    with pytest.raises(InvalidDocumentIdError):
        search_document(query="anything", document_id="../secrets")


def test_search_document_rejects_empty_document(documents_dir: Path) -> None:
    _write(documents_dir, "empty", "   \n  ")

    with pytest.raises(EmptyDocumentError):
        search_document(query="anything", document_id="empty")


def test_search_chunks_uses_reranker_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_candidates = [("candidate one", 0.8), ("candidate two", 0.7), ("candidate three", 0.6)]
    fake_reranked = [("candidate two", 2.0), ("candidate one", 1.0), ("candidate three", 0.0)]

    def fake_retrieve_top_k_chunks(
        query: str, chunks: list[str], top_k: int
    ) -> list[tuple[str, float]]:
        assert top_k == 3
        return fake_candidates

    def fake_rerank_chunks(
        query: str, retrieved_chunks: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        assert retrieved_chunks == fake_candidates
        return fake_reranked

    monkeypatch.setattr(pipeline, "retrieve_top_k_chunks", fake_retrieve_top_k_chunks)
    monkeypatch.setattr(pipeline, "rerank_chunks", fake_rerank_chunks)

    results = search_chunks(
        "İade süresi nedir?", ["irrelevant"], top_k=2, candidate_k=3, use_reranker=True
    )

    assert results == [("candidate two", 2.0), ("candidate one", 1.0)]


def test_search_chunks_never_retrieves_fewer_candidates_than_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, int] = {}

    def fake_retrieve_top_k_chunks(
        query: str, chunks: list[str], top_k: int
    ) -> list[tuple[str, float]]:
        received["top_k"] = top_k
        return []

    monkeypatch.setattr(pipeline, "retrieve_top_k_chunks", fake_retrieve_top_k_chunks)
    monkeypatch.setattr(pipeline, "rerank_chunks", lambda query, retrieved_chunks: [])

    search_chunks("q", ["c"], top_k=4, candidate_k=2, use_reranker=True)

    assert received["top_k"] == 4


def test_search_document_candidate_k_defaults_to_at_least_top_k(
    documents_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(documents_dir, "doc", "some text")
    received: dict[str, object] = {}

    def fake_search_chunks(query: str, chunks: list[str], **kwargs: Any) -> list[tuple[str, float]]:
        received.update(kwargs)
        return []

    monkeypatch.setattr(pipeline, "search_chunks", fake_search_chunks)

    search_document(query="q", document_id="doc", top_k=8, use_reranker=True)

    assert received == {"top_k": 8, "use_reranker": True, "candidate_k": 8}


@pytest.mark.slow
def test_search_document_multilingual_with_real_models(documents_dir: Path) -> None:
    _write(
        documents_dir,
        "support",
        "Customers can contact support by email. "
        "The return period is 14 days. "
        "Our office is located in Berlin.",
    )

    results = search_document(
        query="Müşteri desteğine nasıl ulaşabilirim?",
        document_id="support",
        top_k=1,
        use_reranker=True,
        settings=Settings(chunk_size=6),
    )

    assert "support" in results[0][0].lower()
