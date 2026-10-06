import json
import logging
import time
from typing import Any

import httpx
import ollama
import pytest

from conftest import FakeOllamaClient
from multilingual_support_copilot import config, llm_client
from multilingual_support_copilot.errors import LLMResponseError, LLMUnavailableError
from multilingual_support_copilot.llm_client import (
    generate_answer,
    generate_structured_answer,
    strip_think_blocks,
)

# Bound at collection time: conftest swaps the module attribute per test.
from multilingual_support_copilot.llm_client import get_ollama_client as _real_get_ollama_client
from multilingual_support_copilot.models import AnswerResponse

VALID = json.dumps({"answer": "Refunds take 5 days.", "has_evidence": True, "sources": ["c1"]})


# --- strip_think_blocks ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>reasoning</think>Answer", "Answer"),
        ("<think>a</think>One <think>b</think>Two", "One Two"),
        ("<think>\nline 1\nline 2\n</think>\n\nAnswer", "Answer"),
        ("leaked reasoning\n</think>\nAnswer", "Answer"),
        ("Answer <think>unfinished reasoning\nmore", "Answer"),
        ("Plain answer, no tags.", "Plain answer, no tags."),
        ("<THINK>Upper</THINK>Answer", "Answer"),
        ("  padded  ", "padded"),
    ],
    ids=[
        "closed",
        "multiple",
        "multiline",
        "orphan-close",
        "unclosed-open",
        "no-tags",
        "uppercase",
        "whitespace",
    ],
)
def test_strip_think_blocks(raw: str, expected: str) -> None:
    assert strip_think_blocks(raw) == expected


# --- chat parameters ---------------------------------------------------------


def test_generate_answer_passes_think_false_and_model(fake_ollama: FakeOllamaClient) -> None:
    fake_ollama.queue("<think>hmm</think> Hello ")

    assert generate_answer("hi") == "Hello"
    (call,) = fake_ollama.calls
    assert call["think"] is False
    assert call["model"] == config.get_settings().llm_model_name == "qwen3:4b"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert "format" not in call


def test_env_overrides_think_and_model(
    fake_ollama: FakeOllamaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COPILOT_LLM_THINK", "true")
    monkeypatch.setenv("COPILOT_LLM_MODEL_NAME", "other:1b")
    config.get_settings.cache_clear()
    fake_ollama.queue("ok")

    generate_answer("hi")
    assert fake_ollama.calls[0]["think"] is True
    assert fake_ollama.calls[0]["model"] == "other:1b"


# --- structured output & retries ---------------------------------------------


def test_structured_success_first_try(fake_ollama: FakeOllamaClient) -> None:
    fake_ollama.queue(VALID)

    result = generate_structured_answer("q")
    assert result == AnswerResponse(
        answer="Refunds take 5 days.", has_evidence=True, sources=["c1"]
    )
    (call,) = fake_ollama.calls
    assert call["format"] == AnswerResponse.model_json_schema()
    assert call["think"] is False


def test_structured_strips_think_before_parsing(fake_ollama: FakeOllamaClient) -> None:
    fake_ollama.queue(f"<think>let me see</think>{VALID}")
    assert generate_structured_answer("q").answer == "Refunds take 5 days."


def test_structured_retries_after_invalid_json(
    fake_ollama: FakeOllamaClient, caplog: pytest.LogCaptureFixture
) -> None:
    fake_ollama.queue("not json {", VALID)

    with caplog.at_level(logging.WARNING, logger=llm_client.__name__):
        result = generate_structured_answer("q")
    assert result.has_evidence is True
    assert len(fake_ollama.calls) == 2
    assert any("attempt 1/3" in r.getMessage() for r in caplog.records)


def test_structured_retries_after_schema_invalid_json(fake_ollama: FakeOllamaClient) -> None:
    fake_ollama.queue(json.dumps({"answer": "x"}), VALID)

    assert generate_structured_answer("q").sources == ["c1"]
    assert len(fake_ollama.calls) == 2


def test_structured_gives_up_after_max_retries(fake_ollama: FakeOllamaClient) -> None:
    max_retries = config.get_settings().llm_max_retries
    fake_ollama.queue(*["nope"] * (max_retries + 1))

    with pytest.raises(LLMResponseError) as exc_info:
        generate_structured_answer("q")
    err = exc_info.value
    assert err.attempts == max_retries + 1
    assert err.last_error is not None
    assert err.__cause__ is err.last_error
    assert len(fake_ollama.calls) == max_retries + 1


def test_structured_zero_retries_env(
    fake_ollama: FakeOllamaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COPILOT_LLM_MAX_RETRIES", "0")
    config.get_settings.cache_clear()
    fake_ollama.queue("nope")

    with pytest.raises(LLMResponseError) as exc_info:
        generate_structured_answer("q")
    assert exc_info.value.attempts == 1
    assert len(fake_ollama.calls) == 1


# --- error mapping -----------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected_fragments"),
    [
        (ConnectionError("Failed to connect"), ["http://localhost:11434", "ollama serve"]),
        (httpx.ConnectError("refused"), ["http://localhost:11434", "ollama serve"]),
        (httpx.ReadTimeout("timed out"), ["http://localhost:11434", "60 seconds"]),
        (httpx.ConnectTimeout("timed out"), ["http://localhost:11434", "60 seconds"]),
        (
            ollama.ResponseError("model 'qwen3:4b' not found", status_code=404),
            ["ollama pull qwen3:4b"],
        ),
        (ollama.ResponseError("boom", status_code=500), ["HTTP 500", "qwen3:4b"]),
    ],
    ids=["connection", "httpx-connect", "read-timeout", "connect-timeout", "404", "500"],
)
@pytest.mark.parametrize("structured", [False, True], ids=["plain", "structured"])
def test_backend_errors_map_to_unavailable_without_retry(
    fake_ollama: FakeOllamaClient,
    exc: BaseException,
    expected_fragments: list[str],
    structured: bool,
) -> None:
    fake_ollama.queue(exc, VALID, VALID)

    with pytest.raises(LLMUnavailableError) as exc_info:
        if structured:
            generate_structured_answer("q")
        else:
            generate_answer("q")
    message = str(exc_info.value)
    for fragment in expected_fragments:
        assert fragment in message
    assert exc_info.value.__cause__ is exc
    assert len(fake_ollama.calls) == 1


def test_timeout_message_uses_configured_seconds(
    fake_ollama: FakeOllamaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COPILOT_OLLAMA_TIMEOUT_SECONDS", "7.5")
    config.get_settings.cache_clear()
    fake_ollama.queue(httpx.ReadTimeout("timed out"))

    with pytest.raises(LLMUnavailableError, match=r"7\.5 seconds"):
        generate_answer("q")


# --- client construction -----------------------------------------------------


def test_get_ollama_client_uses_settings_and_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_OLLAMA_HOST", "http://ollama.internal:9999")
    monkeypatch.setenv("COPILOT_OLLAMA_TIMEOUT_SECONDS", "12")
    config.get_settings.cache_clear()
    created: list[dict[str, Any]] = []

    class _CapturingClient:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    monkeypatch.setattr(ollama, "Client", _CapturingClient)
    _real_get_ollama_client.cache_clear()
    try:
        first = _real_get_ollama_client()
        second = _real_get_ollama_client()
    finally:
        _real_get_ollama_client.cache_clear()

    assert first is second
    assert len(created) == 1
    assert created[0]["host"].rstrip("/") == "http://ollama.internal:9999"
    assert created[0]["timeout"] == 12.0


# --- live Ollama ---------------------------------------------------------------


@pytest.fixture
def live_ollama() -> None:
    settings = config.get_settings()
    host = str(settings.ollama_host).rstrip("/")
    try:
        tags = httpx.get(f"{host}/api/tags", timeout=3).json()
    except (httpx.HTTPError, ValueError) as err:
        pytest.skip(f"Ollama not reachable at {host}: {err}")
    names = {m.get("name") for m in tags.get("models", [])}
    if settings.llm_model_name not in names:
        pytest.skip(f"Model {settings.llm_model_name!r} not pulled on {host}")


@pytest.mark.slow
def test_live_ollama_structured_and_plain(live_ollama: None) -> None:
    prompt = (
        "Answer the question using only the context. Respond as JSON with fields "
        "answer (string), has_evidence (boolean), sources (list of chunk ids).\n\n"
        "Context:\n[chunk-1] Refunds are processed within 5 business days.\n\n"
        "Question: How long do refunds take?"
    )
    start = time.perf_counter()
    result = generate_structured_answer(prompt)
    structured_seconds = time.perf_counter() - start

    assert isinstance(result, AnswerResponse)
    assert result.answer
    assert "<think>" not in result.answer.lower()

    start = time.perf_counter()
    text = generate_answer("Reply with one short sentence: what colour is the sky?")
    plain_seconds = time.perf_counter() - start

    assert text
    assert "<think>" not in text.lower()
    print(f"\nlive structured: {structured_seconds:.2f}s, plain: {plain_seconds:.2f}s")


def test_structured_keeps_literal_think_tag_inside_json(fake_ollama: FakeOllamaClient) -> None:
    content = '{"answer": "Use the <think> tag", "has_evidence": true, "sources": []}'
    fake_ollama.queue(content)

    result = generate_structured_answer("prompt")

    assert result.answer == "Use the <think> tag"
    assert len(fake_ollama.calls) == 1
