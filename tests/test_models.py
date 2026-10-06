from typing import Any

import pytest
from pydantic import ValidationError

from multilingual_support_copilot.config import get_settings
from multilingual_support_copilot.models import AnswerResponse, QuestionRequest


def test_answer_response_accepts_valid_data() -> None:
    response = AnswerResponse(
        answer="İade süresi 14 gündür.",
        has_evidence=True,
        sources=["The return period is 14 days."],
    )

    assert response.answer == "İade süresi 14 gündür."
    assert response.has_evidence is True
    assert response.sources == ["The return period is 14 days."]


def test_answer_response_rejects_missing_sources() -> None:
    with pytest.raises(ValidationError):
        AnswerResponse(  # type: ignore[call-arg]
            answer="İade süresi 14 gündür.",
            has_evidence=True,
        )


# --- QuestionRequest -----------------------------------------------------------


def _request(**overrides: Any) -> QuestionRequest:
    data: dict[str, Any] = {"query": "What is the return period?", "document_id": "faq"}
    data.update(overrides)
    return QuestionRequest(**data)


def test_question_request_fills_defaults_from_settings() -> None:
    settings = get_settings()

    request = _request()

    assert request.query == "What is the return period?"
    assert request.document_id == "faq"
    assert request.top_k == settings.default_top_k
    assert request.candidate_k == settings.default_candidate_k
    assert request.use_reranker is False


def test_question_request_honours_env_default_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_DEFAULT_TOP_K", "4")
    get_settings.cache_clear()

    request = _request()

    assert request.top_k == 4
    assert request.candidate_k == get_settings().default_candidate_k


def test_question_request_accepts_explicit_values() -> None:
    request = _request(top_k=2, candidate_k=10, use_reranker=True)

    assert (request.top_k, request.candidate_k, request.use_reranker) == (2, 10, True)


@pytest.mark.parametrize(
    "extra",
    [
        {"file_path": "/etc/passwd"},
        {"file_path": "data/sample.txt"},
        {"chunk_size": 50},
        {"overlap": 0},
    ],
)
def test_question_request_rejects_unknown_fields(extra: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _request(**extra)


def test_question_request_requires_document_id() -> None:
    with pytest.raises(ValidationError):
        QuestionRequest(query="hello")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "document_id",
    [
        "../etc/passwd",
        "..",
        "a/b",
        "a\\b",
        "/etc/passwd",
        ".hidden",
        "Upper",
        "",
        "a" * 65,
        "faq.txt",
        "faq\x00",
        "faq\n",
        "\u0430bc",
    ],
)
def test_question_request_rejects_invalid_document_id(document_id: str) -> None:
    with pytest.raises(ValidationError):
        _request(document_id=document_id)


@pytest.mark.parametrize("query", ["", "   ", "\n\t "])
def test_question_request_rejects_blank_query(query: str) -> None:
    with pytest.raises(ValidationError):
        _request(query=query)


def test_question_request_enforces_max_query_chars() -> None:
    limit = get_settings().max_query_chars

    assert len(_request(query="a" * limit).query) == limit
    with pytest.raises(ValidationError, match="query"):
        _request(query="a" * (limit + 1))


@pytest.mark.parametrize("top_k", [0, -1])
def test_question_request_rejects_non_positive_top_k(top_k: int) -> None:
    with pytest.raises(ValidationError, match="top_k"):
        _request(top_k=top_k)


def test_question_request_rejects_top_k_above_max() -> None:
    settings = get_settings()

    assert _request(top_k=settings.max_top_k).top_k == settings.max_top_k
    with pytest.raises(ValidationError, match="top_k"):
        _request(top_k=settings.max_top_k + 1)


def test_question_request_rejects_candidate_k_below_top_k() -> None:
    with pytest.raises(ValidationError, match="candidate_k must be >= top_k"):
        _request(top_k=5, candidate_k=4)


def test_question_request_rejects_candidate_k_above_max() -> None:
    settings = get_settings()

    with pytest.raises(ValidationError, match="candidate_k"):
        _request(top_k=1, candidate_k=settings.max_candidate_k + 1)


def test_question_request_rejects_non_positive_candidate_k() -> None:
    with pytest.raises(ValidationError, match="candidate_k"):
        _request(candidate_k=0)


def test_question_request_raises_default_candidate_k_to_top_k() -> None:
    settings = get_settings()
    top_k = settings.default_candidate_k + 2
    assert top_k <= settings.max_top_k

    request = _request(top_k=top_k)

    assert request.candidate_k == top_k
