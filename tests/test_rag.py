from pathlib import Path

import multilingual_support_copilot.rag as rag_module


def test_answer_question_returns_llm_answer(monkeypatch) -> None:
    fake_chunks = [
        ("The return period is 14 days.", 0.80),
    ]

    def fake_search_document(**kwargs):
        return fake_chunks

    def fake_build_grounded_prompt(**kwargs):
        return "fake grounded prompt"

    def fake_generate_answer(prompt: str) -> str:
        return "14 gün"

    monkeypatch.setattr(
        rag_module,
        "search_document",
        fake_search_document,
    )

    monkeypatch.setattr(
        rag_module,
        "build_grounded_prompt",
        fake_build_grounded_prompt,
    )

    monkeypatch.setattr(
        rag_module,
        "generate_answer",
        fake_generate_answer,
    )

    result = rag_module.answer_question(
        query="İade için kaç günüm var?",
        file_path=Path("fake.txt"),
    )

    assert result == "14 gün"

def test_answer_question_passes_overlap_to_search_document(monkeypatch) -> None:
    received_overlap = None

    def fake_search_document(**kwargs):
        nonlocal received_overlap
        received_overlap = kwargs["overlap"]

        return [
            ("The return period is 14 days.", 0.80),
        ]

    def fake_build_grounded_prompt(**kwargs):
        return "fake prompt"

    def fake_generate_answer(prompt: str) -> str:
        return "14 gün"

    monkeypatch.setattr(
        rag_module,
        "search_document",
        fake_search_document,
    )

    monkeypatch.setattr(
        rag_module,
        "build_grounded_prompt",
        fake_build_grounded_prompt,
    )

    monkeypatch.setattr(
        rag_module,
        "generate_answer",
        fake_generate_answer,
    )

    rag_module.answer_question(
        query="İade için kaç günüm var?",
        file_path=Path("fake.txt"),
        overlap=10,
    )

    assert received_overlap == 10

def test_answer_question_passes_reranker_settings_to_search_document(
    monkeypatch,
) -> None:
    received_use_reranker = None
    received_candidate_k = None

    def fake_search_document(**kwargs):
        nonlocal received_use_reranker, received_candidate_k

        received_use_reranker = kwargs["use_reranker"]
        received_candidate_k = kwargs["candidate_k"]

        return [
            ("The return period is 14 days.", 0.90),
        ]

    def fake_build_grounded_prompt(**kwargs):
        return "fake grounded prompt"

    def fake_generate_answer(prompt: str) -> str:
        return "14 gün"

    monkeypatch.setattr(
        rag_module,
        "search_document",
        fake_search_document,
    )

    monkeypatch.setattr(
        rag_module,
        "build_grounded_prompt",
        fake_build_grounded_prompt,
    )

    monkeypatch.setattr(
        rag_module,
        "generate_answer",
        fake_generate_answer,
    )

    result = rag_module.answer_question(
        query="İade için kaç günüm var?",
        file_path=Path("fake.txt"),
        use_reranker=True,
        candidate_k=5,
    )

    assert received_use_reranker is True
    assert received_candidate_k == 5
    assert result == "14 gün"