"""Thin wrapper around the Ollama chat API with error mapping and structured-output retries."""

import logging
import re
from functools import lru_cache
from typing import Any

import httpx
import ollama
from pydantic import ValidationError

from multilingual_support_copilot.config import get_settings
from multilingual_support_copilot.errors import LLMResponseError, LLMUnavailableError
from multilingual_support_copilot.models import AnswerResponse

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_ORPHAN_CLOSE_RE = re.compile(r"\A.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_OPEN_RE = re.compile(r"<think\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)
_OPEN_TAG_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"</think\s*>", re.IGNORECASE)


@lru_cache(maxsize=1)
def get_ollama_client() -> ollama.Client:
    settings = get_settings()
    return ollama.Client(
        host=str(settings.ollama_host),
        timeout=settings.ollama_timeout_seconds,
    )


def strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning from model output.

    Handles closed blocks, a leading orphan ``...</think>`` (opening tag missing)
    and a trailing ``<think>...`` that was never closed.
    """
    # Leading orphan close: a closing tag that appears before any opening tag.
    close = _CLOSE_TAG_RE.search(text)
    if close is not None:
        opening = _OPEN_TAG_RE.search(text)
        if opening is None or opening.start() > close.start():
            text = _ORPHAN_CLOSE_RE.sub("", text, count=1)
    text = _THINK_BLOCK_RE.sub("", text)
    text = _UNCLOSED_OPEN_RE.sub("", text)
    return text.strip()


def _chat(prompt: str, *, format: dict[str, Any] | None = None) -> str:
    settings = get_settings()
    host = str(settings.ollama_host)
    model = settings.llm_model_name
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "think": settings.llm_think,
    }
    if format is not None:
        kwargs["format"] = format

    try:
        response = get_ollama_client().chat(**kwargs)
    except httpx.TimeoutException as err:
        raise LLMUnavailableError(
            f"Ollama at {host} did not respond within {settings.ollama_timeout_seconds:g} "
            f"seconds (model {model!r}). Increase COPILOT_OLLAMA_TIMEOUT_SECONDS or check "
            "the server load."
        ) from err
    except (ConnectionError, httpx.TransportError) as err:
        raise LLMUnavailableError(
            f"Could not connect to Ollama at {host}. Is it running? Start it with "
            "`ollama serve` or set COPILOT_OLLAMA_HOST."
        ) from err
    except ollama.ResponseError as err:
        if err.status_code == 404:
            raise LLMUnavailableError(
                f"Model {model!r} is not available on Ollama at {host}. Run `ollama pull {model}`."
            ) from err
        raise LLMUnavailableError(
            f"Ollama at {host} returned HTTP {err.status_code} for model {model!r}: {err.error}"
        ) from err

    return strip_think_blocks(response.message.content or "")


def generate_answer(prompt: str) -> str:
    return _chat(prompt)


def generate_structured_answer(prompt: str) -> AnswerResponse:
    """Ask for an ``AnswerResponse``, retrying when the output fails validation.

    ``LLMUnavailableError`` is not retried. After ``1 + llm_max_retries`` invalid
    outputs, raises ``LLMResponseError``.
    """
    attempts = 1 + get_settings().llm_max_retries
    schema = AnswerResponse.model_json_schema()
    last_error: ValidationError | None = None
    for attempt in range(1, attempts + 1):
        content = _chat(prompt, format=schema)
        try:
            return AnswerResponse.model_validate_json(content)
        except ValidationError as err:
            last_error = err
            logger.warning(
                "LLM structured output invalid (attempt %d/%d): %s",
                attempt,
                attempts,
                err.errors(include_url=False)[:3],
            )
    assert last_error is not None
    raise LLMResponseError(
        f"LLM returned invalid structured output after {attempts} attempt(s)",
        attempts=attempts,
        last_error=last_error,
    ) from last_error
