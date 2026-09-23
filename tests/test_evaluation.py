import json
from pathlib import Path

import multilingual_support_copilot.evaluation as evaluation_module


def test_evaluate_retrieval_returns_correct_recall(
    tmp_path: Path,
    monkeypatch,
) -> None:
    document_path = tmp_path / "document.txt"
    queries_path = tmp_path / "queries.json"

    document_path.write_text(
        "The return period is 14 days.\n"
        "Our office is located in Berlin.",
        encoding="utf-8",
    )

    queries = [
        {
            "id": "q1",
            "language": "tr",
            "query": "İade için kaç günüm var?",
            "expected_evidence": "The return period is 14 days.",
        },
        {
            "id": "q2",
            "language": "en",
            "query": "Where is the office?",
            "expected_evidence": "Our office is located in Berlin.",
        },
    ]

    queries_path.write_text(
        json.dumps(queries),
        encoding="utf-8",
    )

    def fake_retrieve_top_k_chunks(query, chunks, top_k):
        if "İade" in query:
            return [
                ("The return period is 14 days.", 0.9),
            ]

        return [
            ("The return period is 14 days.", 0.8),
        ]

    monkeypatch.setattr(
        evaluation_module,
        "retrieve_top_k_chunks",
        fake_retrieve_top_k_chunks,
    )

    recall = evaluation_module.evaluate_retrieval(
        document_path=document_path,
        queries_path=queries_path,
        top_k=1,
    )

    assert recall == 0.5