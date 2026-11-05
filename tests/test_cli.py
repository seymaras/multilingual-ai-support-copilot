import io
import json
from pathlib import Path
from typing import Any

import pytest

import multilingual_support_copilot.cli as cli
import multilingual_support_copilot.rag as rag_module
from multilingual_support_copilot.errors import EmptyDocumentError, LLMUnavailableError
from multilingual_support_copilot.models import AnswerResponse

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def docs(documents_dir: Path) -> Path:
    (documents_dir / "faq.txt").write_text("The return period is 14 days.", encoding="utf-8")
    return documents_dir


@pytest.fixture
def asked(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the RAG call and model warm-up; record every question asked."""
    calls: list[dict[str, Any]] = []

    def fake_answer(query: str, document_id: str, **kwargs: Any) -> AnswerResponse:
        calls.append({"query": query, "document_id": document_id, **kwargs})
        return AnswerResponse(
            answer=f"answer to {query}",
            has_evidence=True,
            sources=["The return period is 14 days."],
        )

    monkeypatch.setattr(rag_module, "answer_question_structured", fake_answer)
    monkeypatch.setattr(cli, "warm_up", lambda *, use_reranker: None)
    return calls


def run(argv: list[str], stdin: str = "") -> tuple[int, str]:
    out = io.StringIO()
    code = cli.main(argv, inp=io.StringIO(stdin), out=out)
    return code, out.getvalue()


def test_single_question_prints_answer_and_sources(docs: Path, asked: list[dict[str, Any]]) -> None:
    code, output = run(["--document", "faq", "How long?"])

    assert code == 0
    assert asked == [{"query": "How long?", "document_id": "faq", "use_reranker": True}]
    assert "Answer:   answer to How long?" in output
    assert "Evidence: yes" in output
    assert "  - The return period is 14 days." in output


def test_no_rerank_flag_disables_reranker(docs: Path, asked: list[dict[str, Any]]) -> None:
    code, output = run(["--document", "faq", "--no-rerank", "How long?"])

    assert code == 0
    assert asked[0]["use_reranker"] is False
    assert "reranking: off" in output


def test_demo_asks_every_question(docs: Path, asked: list[dict[str, Any]], tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [{"language": "tr", "question": "Bir?"}, {"language": "de", "question": "Zwei?"}]
        ),
        encoding="utf-8",
    )

    code, output = run(["--document", "faq", "--demo", "--questions", str(questions)])

    assert code == 0
    assert [call["query"] for call in asked] == ["Bir?", "Zwei?"]
    assert "[1/2] (TR) Bir?" in output
    assert "[2/2] (DE) Zwei?" in output
    assert "2/2 questions answered" in output


def test_demo_stops_when_llm_is_unavailable(
    docs: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def unavailable(query: str, document_id: str, **kwargs: Any) -> AnswerResponse:
        calls.append(query)
        raise LLMUnavailableError("Ollama is not running")

    monkeypatch.setattr(rag_module, "answer_question_structured", unavailable)
    monkeypatch.setattr(cli, "warm_up", lambda *, use_reranker: None)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"question": "a"}, {"question": "b"}]), encoding="utf-8")

    code, output = run(["--document", "faq", "--demo", "--questions", str(questions)])

    assert code == 1
    assert calls == ["a"]
    assert "Error: Ollama is not running" in output


def test_demo_continues_after_other_errors(
    docs: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def flaky(query: str, document_id: str, **kwargs: Any) -> AnswerResponse:
        if query == "bad":
            raise EmptyDocumentError("empty")
        return AnswerResponse(answer="ok", has_evidence=True, sources=[])

    monkeypatch.setattr(rag_module, "answer_question_structured", flaky)
    monkeypatch.setattr(cli, "warm_up", lambda *, use_reranker: None)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"question": "bad"}, {"question": "good"}]), encoding="utf-8")

    code, output = run(["--document", "faq", "--demo", "--questions", str(questions)])

    assert code == 1
    assert "Error: empty" in output
    assert "1/2 questions answered" in output


def test_interactive_skips_blank_lines_and_quits(docs: Path, asked: list[dict[str, Any]]) -> None:
    code, output = run(["--document", "faq"], stdin="\n  \nFirst?\nq\nnever asked\n")

    assert code == 0
    assert [call["query"] for call in asked] == ["First?"]
    assert "answer to First?" in output


def test_interactive_ends_on_eof(docs: Path, asked: list[dict[str, Any]]) -> None:
    code, _ = run(["--document", "faq"], stdin="Only?\n")

    assert code == 0
    assert [call["query"] for call in asked] == ["Only?"]


def test_interactive_reports_errors_and_keeps_going(
    docs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter([LLMUnavailableError("down"), None])

    def sometimes(query: str, document_id: str, **kwargs: Any) -> AnswerResponse:
        error = next(answers)
        if error:
            raise error
        return AnswerResponse(answer="back", has_evidence=True, sources=[])

    monkeypatch.setattr(rag_module, "answer_question_structured", sometimes)
    monkeypatch.setattr(cli, "warm_up", lambda *, use_reranker: None)

    code, output = run(["--document", "faq"], stdin="one\ntwo\nexit\n")

    assert code == 0
    assert "Error: down" in output
    assert "Answer:   back" in output


def test_unknown_document_lists_available_ids(docs: Path, asked: list[dict[str, Any]]) -> None:
    code, output = run(["--document", "missing", "Hi?"])

    assert code == 2
    assert asked == []
    assert "Unknown document 'missing'" in output
    assert "Available: faq" in output


def test_missing_questions_file_is_reported(
    docs: Path, asked: list[dict[str, Any]], tmp_path: Path
) -> None:
    code, output = run(["--document", "faq", "--demo", "--questions", str(tmp_path / "x.json")])

    assert code == 2
    assert "Questions file not found" in output


@pytest.mark.parametrize(
    "content", ["[]", "{}", "not json", '[{"question": ""}]', '[{"language": "tr"}]']
)
def test_invalid_questions_file_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "questions.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        cli.load_demo_questions(path)


def test_question_and_demo_together_is_an_error(docs: Path) -> None:
    with pytest.raises(SystemExit):
        run(["--demo", "Hi?"])


def test_bundled_demo_questions_are_valid() -> None:
    questions = cli.load_demo_questions(DATA_DIR / "demo_questions.json")

    assert len(questions) >= 15
    assert {q.language for q in questions} == {"tr", "de", "en"}
