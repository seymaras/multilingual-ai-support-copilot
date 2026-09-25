import pytest
from pydantic import ValidationError

from multilingual_support_copilot.models import AnswerResponse


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
        AnswerResponse(
            answer="İade süresi 14 gündür.",
            has_evidence=True,
        )