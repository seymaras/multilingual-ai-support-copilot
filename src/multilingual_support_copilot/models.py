from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from multilingual_support_copilot.config import get_settings
from multilingual_support_copilot.documents import DOCUMENT_ID_PATTERN


class AnswerResponse(BaseModel):
    answer: str
    has_evidence: bool
    sources: list[str]


class QuestionRequest(BaseModel):
    """A client question about a server-managed document.

    Chunking parameters are server settings and cannot be chosen by the client.
    Retrieval limits default to, and are bounded by, the current settings.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    document_id: str = Field(pattern=DOCUMENT_ID_PATTERN)
    top_k: int | None = None
    candidate_k: int | None = None
    use_reranker: bool = False

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def _apply_settings(self) -> Self:
        settings = get_settings()

        if len(self.query) > settings.max_query_chars:
            raise ValueError(f"query must be at most {settings.max_query_chars} characters")

        if self.top_k is None:
            self.top_k = settings.default_top_k
        if self.candidate_k is None:
            self.candidate_k = max(settings.default_candidate_k, self.top_k)

        if not 1 <= self.top_k <= settings.max_top_k:
            raise ValueError(f"top_k must be between 1 and {settings.max_top_k}")
        if not 1 <= self.candidate_k <= settings.max_candidate_k:
            raise ValueError(f"candidate_k must be between 1 and {settings.max_candidate_k}")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        return self
