import subprocess
import sys
import types
from typing import Any

import numpy as np
import pytest

from multilingual_support_copilot import config, embeddings, reranker

# Grab the real (lru-cached) getters before the autouse fixture swaps them out.
from multilingual_support_copilot.embeddings import get_embedding_model as _real_get_embedding_model
from multilingual_support_copilot.reranker import get_reranker_model as _real_get_reranker_model

# --- Import hygiene ------------------------------------------------------------


def _heavy_modules_loaded_after_import(*modules: str) -> str:
    imports = ", ".join(modules)
    code = (
        f"import sys, {imports}; "
        "print(','.join(m for m in ('torch', 'sentence_transformers') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_pipeline_import_does_not_load_torch() -> None:
    loaded = _heavy_modules_loaded_after_import(
        "multilingual_support_copilot.retriever",
        "multilingual_support_copilot.embeddings",
        "multilingual_support_copilot.reranker",
        "multilingual_support_copilot.pipeline",
    )
    assert loaded == ""


def test_public_entry_points_do_not_load_torch() -> None:
    loaded = _heavy_modules_loaded_after_import(
        "multilingual_support_copilot.api",
        "multilingual_support_copilot.rag",
        "multilingual_support_copilot.pipeline",
        "multilingual_support_copilot.evaluation",
    )
    assert loaded == ""


# --- Lazy singletons -------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.names: list[str] = []

    def make_class(self) -> type:
        recorder = self

        class _FakeModel:
            def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
                recorder.names.append(model_name)

        return _FakeModel


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> dict[str, _Recorder]:
    recorders = {"SentenceTransformer": _Recorder(), "CrossEncoder": _Recorder()}
    module = types.ModuleType("sentence_transformers")
    for name, recorder in recorders.items():
        setattr(module, name, recorder.make_class())
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return recorders


def test_embedding_model_is_constructed_once_with_settings_name(
    fake_sentence_transformers: dict[str, _Recorder],
) -> None:
    first = _real_get_embedding_model()
    second = _real_get_embedding_model()

    assert first is second
    assert fake_sentence_transformers["SentenceTransformer"].names == [
        config.get_settings().embedding_model_name
    ]


def test_embedding_model_name_env_override(
    fake_sentence_transformers: dict[str, _Recorder], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COPILOT_EMBEDDING_MODEL_NAME", "acme/embedder")
    config.get_settings.cache_clear()

    _real_get_embedding_model()

    assert fake_sentence_transformers["SentenceTransformer"].names == ["acme/embedder"]


def test_reranker_model_is_constructed_once_with_settings_name(
    fake_sentence_transformers: dict[str, _Recorder],
) -> None:
    first = _real_get_reranker_model()
    second = _real_get_reranker_model()

    assert first is second
    assert fake_sentence_transformers["CrossEncoder"].names == [
        config.get_settings().reranker_model_name
    ]


def test_reranker_model_name_env_override(
    fake_sentence_transformers: dict[str, _Recorder], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COPILOT_RERANKER_MODEL_NAME", "acme/reranker")
    config.get_settings.cache_clear()

    _real_get_reranker_model()

    assert fake_sentence_transformers["CrossEncoder"].names == ["acme/reranker"]


# --- embed_texts -----------------------------------------------------------------


def test_embed_texts_empty_returns_empty_array_without_model() -> None:
    # The autouse fixture makes get_embedding_model raise, so this also proves
    # the model is never touched for empty input.
    result = embeddings.embed_texts([])

    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_embed_texts_returns_normalized_rows(fake_embedder: Any) -> None:
    result = embeddings.embed_texts(["return period", "office in Berlin", "password reset"])

    assert result.shape == (3, fake_embedder.dim)
    assert result.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, rtol=1e-5)
    assert len(fake_embedder.calls) == 1


# --- rerank_chunks -----------------------------------------------------------------


class FakeReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool = False) -> Any:
        self.calls.append(list(pairs))
        return np.array([self.scores[chunk] for _, chunk in pairs], dtype=np.float32)


@pytest.fixture
def fake_reranker(monkeypatch: pytest.MonkeyPatch) -> FakeReranker:
    model = FakeReranker({"a": 0.1, "b": 2.5, "c": -1.0})
    monkeypatch.setattr(reranker, "get_reranker_model", lambda: model)
    return model


def test_rerank_sorts_by_cross_encoder_score(fake_reranker: FakeReranker) -> None:
    results = reranker.rerank_chunks("q", [("a", 0.9), ("b", 0.5), ("c", 0.7)])

    assert [chunk for chunk, _ in results] == ["b", "a", "c"]
    assert all(type(score) is float for _, score in results)
    assert results[0][1] == pytest.approx(2.5)
    assert fake_reranker.calls == [[("q", "a"), ("q", "b"), ("q", "c")]]


def test_rerank_empty_input_skips_model(fake_reranker: FakeReranker) -> None:
    assert reranker.rerank_chunks("q", []) == []
    assert fake_reranker.calls == []


# --- Real models -------------------------------------------------------------------


@pytest.mark.slow
def test_real_reranker_ranks_relevant_chunk_first_for_turkish_query() -> None:
    candidates = [
        ("Customers can reset their password by email.", 0.0),
        ("Our office is located in Berlin.", 0.0),
        ("The return period is 14 days.", 0.0),
    ]

    results = reranker.rerank_chunks("İade için kaç günüm var?", candidates)

    assert results[0][0] == "The return period is 14 days."
    assert all(type(score) is float for _, score in results)


@pytest.mark.slow
def test_real_embeddings_are_normalized() -> None:
    result = embeddings.embed_texts(["İade süresi", "Return period"])

    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, rtol=1e-5)
