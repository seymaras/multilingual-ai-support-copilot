"""Ask questions about a document from the terminal.

Three modes:

- ``copilot-ask``                    interactive: type questions until ``q``
- ``copilot-ask "question"``         answer one question and exit
- ``copilot-ask --demo``             run the sample questions in ``data/demo_questions.json``
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from multilingual_support_copilot import embeddings, rag, reranker
from multilingual_support_copilot.documents import get_document_registry
from multilingual_support_copilot.errors import CopilotError, LLMUnavailableError
from multilingual_support_copilot.models import AnswerResponse

DEFAULT_DOCUMENT_ID = "evaluation_document"
DEFAULT_QUESTIONS_FILE = Path("data/demo_questions.json")
QUIT_COMMANDS = frozenset({"q", "quit", "exit"})


@dataclass(frozen=True)
class DemoQuestion:
    language: str
    question: str


def load_demo_questions(path: Path) -> list[DemoQuestion]:
    """Read ``[{"language": "tr", "question": "..."}, ...]`` from ``path``."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise ValueError(f"Questions file not found: {path}") from err
    except json.JSONDecodeError as err:
        raise ValueError(f"Questions file is not valid JSON: {path} ({err})") from err

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Questions file must contain a non-empty JSON list: {path}")
    questions: list[DemoQuestion] = []
    for index, item in enumerate(raw, start=1):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("question"), str)
            or not item["question"].strip()
        ):
            raise ValueError(f"Entry {index} in {path} needs a non-empty 'question' string")
        questions.append(
            DemoQuestion(language=str(item.get("language", "?")), question=item["question"])
        )
    return questions


def warm_up(*, use_reranker: bool) -> None:
    """Load the models up front so the first question is not unexpectedly slow."""
    embeddings.get_embedding_model()
    if use_reranker:
        reranker.get_reranker_model()


def print_answer(response: AnswerResponse, seconds: float, out: TextIO) -> None:
    print(f"Answer:   {response.answer}", file=out)
    print(f"Evidence: {'yes' if response.has_evidence else 'no'}", file=out)
    if response.sources:
        print("Sources:", file=out)
        for source in response.sources:
            print(f"  - {source}", file=out)
    print(f"Time:     {seconds:.1f} s", file=out)


def ask(question: str, *, document_id: str, use_reranker: bool, out: TextIO) -> None:
    """Answer ``question`` and print the result. Domain errors propagate to the caller."""
    started = time.perf_counter()
    response = rag.answer_question_structured(question, document_id, use_reranker=use_reranker)
    print_answer(response, time.perf_counter() - started, out)


def run_demo(
    questions: Sequence[DemoQuestion], *, document_id: str, use_reranker: bool, out: TextIO
) -> int:
    answered = 0
    started = time.perf_counter()
    for number, item in enumerate(questions, start=1):
        print(f"\n[{number}/{len(questions)}] ({item.language.upper()}) {item.question}", file=out)
        try:
            ask(item.question, document_id=document_id, use_reranker=use_reranker, out=out)
        except LLMUnavailableError as err:
            # Every later question would fail the same way.
            print(f"Error: {err}", file=out)
            return 1
        except CopilotError as err:
            print(f"Error: {err}", file=out)
            continue
        answered += 1

    total = time.perf_counter() - started
    print(f"\n{answered}/{len(questions)} questions answered in {total:.1f} s", file=out)
    return 0 if answered == len(questions) else 1


def run_interactive(*, document_id: str, use_reranker: bool, inp: TextIO, out: TextIO) -> int:
    print("Ask a question in Turkish, German or English. Type 'q' to quit.", file=out)
    while True:
        print("\n> ", end="", file=out, flush=True)
        try:
            line = inp.readline()
        except KeyboardInterrupt:
            line = ""
        if not line:  # EOF or Ctrl+C
            print(file=out)
            return 0
        question = line.strip()
        if not question:
            continue
        if question.casefold() in QUIT_COMMANDS:
            return 0
        try:
            ask(question, document_id=document_id, use_reranker=use_reranker, out=out)
        except CopilotError as err:
            print(f"Error: {err}", file=out)


def _check_document(document_id: str) -> str | None:
    """Return an error message when ``document_id`` cannot be used, else ``None``."""
    registry = get_document_registry()
    try:
        registry.resolve(document_id)
    except CopilotError:
        available = ", ".join(registry.list_ids()) or "none"
        return f"Unknown document {document_id!r} in {registry.base_dir}. Available: {available}"
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="copilot-ask",
        description="Ask questions about a document. Without arguments, starts an interactive "
        "session.",
    )
    parser.add_argument("question", nargs="?", help="answer this one question and exit")
    parser.add_argument(
        "--demo", action="store_true", help="run the sample questions from --questions"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_FILE,
        help=f"sample questions for --demo (default: {DEFAULT_QUESTIONS_FILE})",
    )
    parser.add_argument(
        "--document",
        default=DEFAULT_DOCUMENT_ID,
        help=f"document ID to answer from (default: {DEFAULT_DOCUMENT_ID})",
    )
    parser.add_argument("--no-rerank", action="store_true", help="skip cross-encoder reranking")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    inp: TextIO | None = None,
    out: TextIO | None = None,
) -> int:
    inp = inp or sys.stdin
    out = out or sys.stdout
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.demo and args.question:
        parser.error("give either a question or --demo, not both")
    use_reranker = not args.no_rerank

    questions: list[DemoQuestion] = []
    if args.demo:
        try:
            questions = load_demo_questions(args.questions)
        except ValueError as err:
            print(f"Error: {err}", file=out)
            return 2

    problem = _check_document(args.document)
    if problem:
        print(f"Error: {problem}", file=out)
        return 2

    rerank_label = "on" if use_reranker else "off"
    print(f"Document: {args.document} | reranking: {rerank_label}", file=out)
    print("Loading models (the first start can take up to a minute)...", file=out, flush=True)
    warm_up(use_reranker=use_reranker)

    if args.demo:
        return run_demo(questions, document_id=args.document, use_reranker=use_reranker, out=out)
    if args.question:
        print(f"\nQuestion: {args.question}", file=out)
        try:
            ask(args.question, document_id=args.document, use_reranker=use_reranker, out=out)
        except CopilotError as err:
            print(f"Error: {err}", file=out)
            return 1
        return 0
    return run_interactive(document_id=args.document, use_reranker=use_reranker, inp=inp, out=out)


if __name__ == "__main__":
    raise SystemExit(main())
