from pathlib import Path

import pytest
from pydantic import ValidationError

from multilingual_support_copilot.config import Settings, get_settings


def test_defaults() -> None:
    settings = Settings()

    assert settings.llm_model_name == "qwen3:4b"
    assert settings.llm_think is False
    assert str(settings.ollama_host) == "http://localhost:11434/"
    assert settings.chunk_size == 50
    assert settings.chunk_overlap == 0
    assert settings.default_top_k == 3
    assert settings.default_candidate_k == 5
    assert settings.documents_dir == Path("data")


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_LLM_MODEL_NAME", "llama3.2:3b")
    monkeypatch.setenv("COPILOT_OLLAMA_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("COPILOT_CHUNK_SIZE", "120")

    settings = Settings()

    assert settings.llm_model_name == "llama3.2:3b"
    assert settings.ollama_timeout_seconds == 5.0
    assert settings.chunk_size == 120


def test_reads_dotenv_from_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("COPILOT_DEFAULT_TOP_K=4\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert Settings().default_top_k == 4


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


@pytest.mark.parametrize(
    "overrides",
    [
        {"default_top_k": 6, "default_candidate_k": 5},
        {"chunk_size": 10, "chunk_overlap": 10},
        {"chunk_size": 600},
        {"default_top_k": 30},
        {"default_candidate_k": 60},
        {"max_top_k": 60, "max_candidate_k": 50},
        {"ollama_timeout_seconds": 0},
        {"chunk_size": 0},
        {"llm_max_retries": -1},
        {"ollama_host": "not a url"},
    ],
)
def test_invalid_settings_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings(**overrides)  # type: ignore[arg-type]
