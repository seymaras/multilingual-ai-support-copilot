"""Shared fixtures.

Every fast test runs with:
- a clean environment (no ``COPILOT_*`` variables, no ``.env`` from the repo root),
- fresh settings / model / client caches,
- real model and Ollama getters replaced by functions that fail loudly.

Tests marked ``@pytest.mark.slow`` keep the real getters.
"""

import os
import re
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import ollama
import pytest

from multilingual_support_copilot import config, embeddings, llm_client, reranker

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

_CACHED_GETTERS = (
    config.get_settings,
    embeddings.get_embedding_model,
    reranker.get_reranker_model,
    llm_client.get_ollama_client,
)


def _clear_caches() -> None:
    for getter in _CACHED_GETTERS:
        getter.cache_clear()


@pytest.fixture(autouse=True)
def _isolated_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    for name in list(os.environ):
        if name.startswith("COPILOT_"):
            monkeypatch.delenv(name)
    # Settings reads ``.env`` relative to the cwd; run from an empty dir.
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    _clear_caches()
    yield
    _clear_caches()


def _forbidden(name: str) -> Any:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{name} called in a fast test; use a fake or mark the test slow")

    return _raise


@pytest.fixture(autouse=True)
def _block_real_models(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("slow"):
        return
    monkeypatch.setattr(embeddings, "get_embedding_model", _forbidden("get_embedding_model"))
    monkeypatch.setattr(reranker, "get_reranker_model", _forbidden("get_reranker_model"))
    monkeypatch.setattr(llm_client, "get_ollama_client", _forbidden("get_ollama_client"))


# --- Fake embedder -----------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class FakeEmbeddingModel:
    """Deterministic bag-of-words hashing embedder (no ML, no downloads).

    Texts sharing more words get a higher cosine similarity, which is enough
    to make ranking tests meaningful.
    """

    dim = 256

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.calls.append(list(sentences))
        matrix = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for row, text in enumerate(sentences):
            for token in _TOKEN_RE.findall(text.casefold()):
                matrix[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix


@pytest.fixture
def fake_embedder(_block_real_models: None, monkeypatch: pytest.MonkeyPatch) -> FakeEmbeddingModel:
    model = FakeEmbeddingModel()
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: model)
    return model


# --- Fake Ollama -------------------------------------------------------------


class FakeOllamaClient:
    """Records ``chat`` kwargs and replays queued responses or exceptions."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._queue: list[str | BaseException] = []

    def queue(self, *items: str | BaseException) -> None:
        self._queue.extend(items)

    def chat(self, **kwargs: Any) -> ollama.ChatResponse:
        self.calls.append(kwargs)
        if not self._queue:
            raise AssertionError("FakeOllamaClient.chat called with an empty response queue")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return ollama.ChatResponse(
            model=kwargs.get("model", "fake"),
            message=ollama.Message(role="assistant", content=item),
        )


@pytest.fixture
def fake_ollama(_block_real_models: None, monkeypatch: pytest.MonkeyPatch) -> FakeOllamaClient:
    client = FakeOllamaClient()
    monkeypatch.setattr(llm_client, "get_ollama_client", lambda: client)
    return client


# --- Documents ---------------------------------------------------------------


@pytest.fixture
def documents_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty documents directory wired into Settings via ``COPILOT_DOCUMENTS_DIR``."""
    directory = tmp_path / "documents"
    directory.mkdir()
    monkeypatch.setenv("COPILOT_DOCUMENTS_DIR", str(directory))
    config.get_settings.cache_clear()
    return directory
