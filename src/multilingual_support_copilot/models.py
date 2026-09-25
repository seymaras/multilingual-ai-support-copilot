from pydantic import BaseModel


class AnswerResponse(BaseModel):
    answer: str
    has_evidence: bool
    sources: list[str]

    