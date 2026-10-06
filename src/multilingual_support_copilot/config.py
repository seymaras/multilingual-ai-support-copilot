"""Application settings, loaded from environment variables (prefix ``COPILOT_``) or ``.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import AnyHttpUrl, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COPILOT_",
        env_file=".env",
        extra="ignore",
    )

    # Models
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    llm_model_name: str = "qwen3:4b"

    # Ollama
    ollama_host: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    ollama_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_think: bool = False

    # Documents (referenced by server-managed IDs only)
    documents_dir: Path = Path("data")

    # Chunking (server-side only; never chosen by the client)
    chunk_size: int = Field(default=50, ge=1)
    max_chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=0, ge=0)

    # Retrieval
    default_top_k: int = Field(default=3, ge=1)
    max_top_k: int = Field(default=20, ge=1)
    default_candidate_k: int = Field(default=5, ge=1)
    max_candidate_k: int = Field(default=50, ge=1)
    max_query_chars: int = Field(default=2000, ge=1)

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.chunk_size > self.max_chunk_size:
            raise ValueError("chunk_size must be <= max_chunk_size")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k must be <= max_top_k")
        if self.default_candidate_k < self.default_top_k:
            raise ValueError("default_candidate_k must be >= default_top_k")
        if self.default_candidate_k > self.max_candidate_k:
            raise ValueError("default_candidate_k must be <= max_candidate_k")
        if self.max_candidate_k < self.max_top_k:
            raise ValueError("max_candidate_k must be >= max_top_k")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
