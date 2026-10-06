from typing import Any

import pytest

from multilingual_support_copilot.retriever import retrieve_top_k_chunks

CHUNKS = [
    "Customers can reset their password by email.",
    "The return period is 14 days.",
    "Returned products must be unused.",
    "Our office is located in Berlin.",
]


def test_ranks_chunks_by_word_overlap(fake_embedder: Any) -> None:
    results = retrieve_top_k_chunks("what is the return period in days", CHUNKS, top_k=4)

    assert results[0][0] == "The return period is 14 days."
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert all(type(score) is float for score in scores)
    assert {chunk for chunk, _ in results} == set(CHUNKS)


def test_truncates_to_top_k(fake_embedder: Any) -> None:
    results = retrieve_top_k_chunks("return period", CHUNKS, top_k=2)

    assert len(results) == 2
    assert results[0][1] >= results[1][1]


def test_top_k_larger_than_chunks_returns_all(fake_embedder: Any) -> None:
    results = retrieve_top_k_chunks("return period", CHUNKS[:2], top_k=10)

    assert len(results) == 2


def test_empty_chunks_return_empty_list_without_encoding(fake_embedder: Any) -> None:
    assert retrieve_top_k_chunks("anything", []) == []
    assert fake_embedder.calls == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises(fake_embedder: Any, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        retrieve_top_k_chunks("return period", CHUNKS, top_k=top_k)


def test_ties_keep_original_order(fake_embedder: Any) -> None:
    chunks = ["alpha beta", "gamma delta", "alpha beta"]

    results = retrieve_top_k_chunks("", chunks, top_k=3)

    assert [chunk for chunk, _ in results] == chunks


def test_embeds_query_and_chunks_in_single_encode_call(fake_embedder: Any) -> None:
    retrieve_top_k_chunks("return period", CHUNKS, top_k=2)

    assert fake_embedder.calls == [["return period", *CHUNKS]]


@pytest.mark.slow
def test_turkish_query_retrieves_english_chunk() -> None:
    chunks = [
        "Customers can reset their password by email.",
        "The return period is 14 days.",
        "Our office is located in Berlin.",
    ]

    results = retrieve_top_k_chunks("İade için kaç günüm var?", chunks, top_k=1)

    assert results[0][0] == "The return period is 14 days."
    assert isinstance(results[0][1], float)


@pytest.mark.slow
def test_turkish_query_returns_ranked_results() -> None:
    results = retrieve_top_k_chunks("İade hakkında bilgi istiyorum.", CHUNKS, top_k=2)

    assert len(results) == 2
    assert results[0][1] >= results[1][1]
