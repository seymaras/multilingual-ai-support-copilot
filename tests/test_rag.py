from typing import Any

import pytest

import multilingual_support_copilot.rag as rag_module
from multilingual_support_copilot.errors import LLMUnavailableError
from multilingual_support_copilot.models import AnswerResponse, QuestionRequest

FAKE_CHUNKS = [("The return period is 14 days.", 0.80)]


@pytest.fixture
def search_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_search_document(**kwargs: Any) -> list[tuple[str, float]]:
        calls.append(kwargs)
        return FAKE_CHUNKS

    monkeypatch.setattr(rag_module, "search_document", fake_search_document)
    return calls


def _forbid_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(prompt: str) -> Any:
        raise AssertionError("LLM must not be called without context")

    monkeypatch.setattr(rag_module, "generate_answer", fail)
    monkeypatch.setattr(rag_module, "generate_structured_answer", fail)


def test_answer_question_returns_llm_answer(
    search_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[str] = []

    def fake_generate_answer(prompt: str) -> str:
        prompts.append(prompt)
        return "14 gün"

    monkeypatch.setattr(rag_module, "generate_answer", fake_generate_answer)

    result = rag_module.answer_question(query="İade için kaç günüm var?", document_id="faq")

    assert result == "14 gün"
    assert "The return period is 14 days." in prompts[0]
    assert "İade için kaç günüm var?" in prompts[0]


def test_answer_question_passes_retrieval_options(
    search_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag_module, "generate_answer", lambda prompt: "14 gün")

    rag_module.answer_question(
        query="İade için kaç günüm var?",
        document_id="faq",
        top_k=2,
        use_reranker=True,
        candidate_k=5,
    )

    assert search_calls == [
        {
            "query": "İade için kaç günüm var?",
            "document_id": "faq",
            "top_k": 2,
            "use_reranker": True,
            "candidate_k": 5,
        }
    ]


def test_answer_question_structured_returns_answer_response(
    search_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_response = AnswerResponse(
        answer="14 gün", has_evidence=True, sources=["The return period is 14 days."]
    )
    monkeypatch.setattr(rag_module, "generate_structured_answer", lambda prompt: fake_response)

    result = rag_module.answer_question_structured(
        query="İade için kaç günüm var?", document_id="faq"
    )

    assert result == fake_response


def test_empty_retrieval_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag_module, "search_document", lambda **kwargs: [])
    _forbid_llm(monkeypatch)

    structured = rag_module.answer_question_structured(query="?", document_id="faq")
    plain = rag_module.answer_question(query="?", document_id="faq")

    assert structured == AnswerResponse(
        answer=rag_module.NO_CONTEXT_ANSWER, has_evidence=False, sources=[]
    )
    assert plain == rag_module.NO_CONTEXT_ANSWER


def test_llm_errors_propagate(
    search_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(prompt: str) -> AnswerResponse:
        raise LLMUnavailableError("Ollama is not reachable")

    monkeypatch.setattr(rag_module, "generate_structured_answer", unavailable)

    with pytest.raises(LLMUnavailableError):
        rag_module.answer_question_structured(query="q", document_id="faq")


def test_answer_request_uses_validated_request(
    search_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_response = AnswerResponse(answer="ok", has_evidence=True, sources=[])
    monkeypatch.setattr(rag_module, "generate_structured_answer", lambda prompt: fake_response)
    request = QuestionRequest(query="Wo ist der Hauptsitz?", document_id="faq", top_k=2)

    result = rag_module.answer_request(request)

    assert result == fake_response
    assert search_calls[0]["document_id"] == "faq"
    assert search_calls[0]["top_k"] == 2
    assert search_calls[0]["candidate_k"] == request.candidate_k
