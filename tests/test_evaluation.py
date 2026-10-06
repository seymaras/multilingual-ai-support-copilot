import json
from pathlib import Path
from typing import Any

import pytest

from multilingual_support_copilot import evaluation, pipeline
from multilingual_support_copilot.evaluation import (
    EvalQuery,
    chunk_contains_span,
    evaluate,
    first_relevant_rank,
    load_queries,
    normalize,
    validate_spans,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = REPO_ROOT / "data" / "evaluation_document.txt"
QUERIES_PATH = REPO_ROOT / "data" / "evaluation_queries.json"


def _q(query_id: str, *spans: str, query: str | None = None) -> EvalQuery:
    return EvalQuery(
        id=query_id, language="en", query=query or query_id, expected_answer_spans=spans
    )


def _write_queries(path: Path, items: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


class FakeSearch:
    """Returns a fixed ranking per query and records call arguments."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        query: str,
        chunks: list[str],
        *,
        top_k: int,
        use_reranker: bool = False,
        candidate_k: int | None = None,
    ) -> list[tuple[str, float]]:
        self.calls.append(
            {
                "query": query,
                "chunks": list(chunks),
                "top_k": top_k,
                "use_reranker": use_reranker,
                "candidate_k": candidate_k,
            }
        )
        ranking = self.rankings.get(query, chunks)
        return [(c, 1.0 / (i + 1)) for i, c in enumerate(ranking[:top_k])]


# --- normalization / span matching ------------------------------------------


def test_normalize_collapses_whitespace_and_casefolds() -> None:
    assert normalize("  The  Return\nPeriod\tIS ") == "the return period is"
    assert normalize("STRASSE") == normalize("strasse")


def test_chunk_contains_span_ignores_case_and_whitespace() -> None:
    chunk = "The return period is 14 days. Returned products must be unused."
    assert chunk_contains_span(chunk, "RETURN period  is\n14 days")
    assert not chunk_contains_span(chunk, "return period is 30 days")


def test_validate_spans_names_missing_span() -> None:
    queries = [_q("q1", "return period is 14 days"), _q("q2", "office is in Paris")]
    with pytest.raises(ValueError, match=r"q2: 'office is in Paris'"):
        validate_spans("The return period is 14 days.", queries)


def test_validate_spans_accepts_present_spans() -> None:
    validate_spans("The Return\n period is 14 days.", [_q("q1", "return period is 14 days")])


# --- loading -----------------------------------------------------------------


def test_load_queries_happy_path(tmp_path: Path) -> None:
    path = _write_queries(
        tmp_path / "q.json",
        [
            {"id": "a", "language": "tr", "query": "x?", "expected_answer_spans": ["s1", "s2"]},
            {"id": "b", "language": "de", "query": "y?", "expected_answer_spans": ["s3"]},
        ],
    )
    queries = load_queries(path)
    assert queries == [
        EvalQuery("a", "tr", "x?", ("s1", "s2")),
        EvalQuery("b", "de", "y?", ("s3",)),
    ]


@pytest.mark.parametrize(
    "item",
    [
        {"id": "a", "language": "en", "query": "x"},
        {"id": "a", "language": "en", "query": "x", "expected_answer_spans": []},
        {"id": "a", "language": "en", "query": "x", "expected_answer_spans": [""]},
        {"id": "a", "language": "en", "query": "x", "expected_answer_spans": "s"},
    ],
)
def test_load_queries_rejects_missing_spans(tmp_path: Path, item: dict[str, Any]) -> None:
    path = _write_queries(tmp_path / "q.json", [item])
    with pytest.raises(ValueError, match="expected_answer_spans"):
        load_queries(path)


def test_load_queries_rejects_duplicate_ids(tmp_path: Path) -> None:
    item = {"id": "a", "language": "en", "query": "x", "expected_answer_spans": ["s"]}
    path = _write_queries(tmp_path / "q.json", [item, item])
    with pytest.raises(ValueError, match="duplicate"):
        load_queries(path)


def test_repo_query_spans_occur_in_repo_document() -> None:
    queries = load_queries(QUERIES_PATH)
    assert len(queries) >= 14
    assert any(len(q.expected_answer_spans) > 1 for q in queries)
    validate_spans(DOCUMENT_PATH.read_text(encoding="utf-8"), queries)


# --- metric math -------------------------------------------------------------


def test_first_relevant_rank() -> None:
    retrieved = ["alpha", "beta gamma", "delta"]
    assert first_relevant_rank(retrieved, ["GAMMA"]) == 2
    assert first_relevant_rank(retrieved, ["zeta", "delta"]) == 3
    assert first_relevant_rank(retrieved, ["zeta"]) is None


# One word per chunk with chunk_size=1 keeps the fixture readable.
DOC = "alpha beta gamma delta epsilon"


def test_hit_at_k_and_mrr(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSearch(
        {
            "first": ["alpha", "beta", "gamma"],
            "third": ["delta", "epsilon", "beta"],
            "miss": ["alpha", "delta", "epsilon"],
        }
    )
    monkeypatch.setattr(pipeline, "search_chunks", fake)
    queries = [
        _q("q1", "alpha", query="first"),
        _q("q2", "beta", query="third"),
        _q("q3", "gamma", query="miss"),
    ]

    metrics, records = evaluate(DOC, queries, chunk_size=1, overlap=0, ks=(1, 3))

    assert metrics.hit_at_k[1] == pytest.approx(1 / 3)
    assert metrics.hit_at_k[3] == pytest.approx(2 / 3)
    assert metrics.recall_at_k[3] == pytest.approx(2 / 3)
    assert metrics.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert metrics.mrr_cutoff == 3
    assert metrics.num_queries == 3
    assert metrics.num_chunks == 5
    assert metrics.unreachable_queries == []
    assert [r["first_relevant_rank"] for r in records] == [1, 3, None]
    assert records[0]["retrieved"][0]["preview"] == "alpha"
    assert set(records[0]) >= {"id", "language", "query", "first_relevant_rank", "retrieved"}


def test_recall_for_multi_span_query_with_one_span_retrieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "search_chunks", FakeSearch({"q": ["alpha", "delta"]}))
    metrics, _ = evaluate(
        DOC, [_q("q1", "alpha", "beta", query="q")], chunk_size=1, overlap=0, ks=(2,)
    )
    assert metrics.hit_at_k[2] == 1.0
    assert metrics.recall_at_k[2] == 0.5


def test_span_straddling_chunk_boundary_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "search_chunks", FakeSearch({}))
    queries = [_q("inside", "alpha beta"), _q("straddles", "beta gamma")]
    metrics, _ = evaluate(DOC, queries, chunk_size=2, overlap=0, ks=(1, 3))
    # Chunks: "alpha beta" | "gamma delta" | "epsilon"
    assert metrics.num_chunks == 3
    assert metrics.unreachable_queries == ["straddles"]
    assert metrics.recall_at_k[3] == 0.5


def test_evaluate_validates_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "search_chunks", FakeSearch({}))
    with pytest.raises(ValueError, match="omega"):
        evaluate(DOC, [_q("q1", "omega")], chunk_size=1, overlap=0)


def test_evaluate_passes_parameters_through(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_calls: list[tuple[str, int, int]] = []
    from multilingual_support_copilot.text_chunker import chunk_text as real_chunk_text

    def spy_chunk_text(text: str, chunk_size: int = 50, overlap: int = 0) -> list[str]:
        chunk_calls.append((text, chunk_size, overlap))
        return real_chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    fake = FakeSearch({})
    monkeypatch.setattr(evaluation, "chunk_text", spy_chunk_text)
    monkeypatch.setattr(pipeline, "search_chunks", fake)

    evaluate(
        DOC,
        [_q("q1", "alpha"), _q("q2", "beta")],
        chunk_size=3,
        overlap=1,
        ks=(1, 10, 3),
        use_reranker=True,
        candidate_k=7,
    )

    assert chunk_calls == [(DOC, 3, 1)]
    # Per query: one max-k call (rerank pool 10), plus k=1 and k=3 re-runs because
    # production reranks a pool of max(candidate_k=7, k) = 7 for those cut-offs.
    assert [c["top_k"] for c in fake.calls] == [10, 1, 3, 10, 1, 3]
    for call in fake.calls:
        assert call["use_reranker"] is True
        assert call["candidate_k"] == 7
        assert call["chunks"] == ["alpha beta gamma", "gamma delta epsilon"]


def test_evaluate_dense_retrieves_once_per_query(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSearch({})
    monkeypatch.setattr(pipeline, "search_chunks", fake)

    evaluate(DOC, [_q("q1", "alpha"), _q("q2", "beta")], chunk_size=3, overlap=1, ks=(1, 3, 10))

    assert [c["top_k"] for c in fake.calls] == [10, 10]


def test_evaluate_reuses_max_k_call_when_rerank_pool_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSearch({})
    monkeypatch.setattr(pipeline, "search_chunks", fake)

    evaluate(
        DOC, [_q("q1", "alpha")], chunk_size=3, overlap=1, ks=(1, 3, 5),
        use_reranker=True, candidate_k=5,
    )  # fmt: skip

    assert [c["top_k"] for c in fake.calls] == [5]


# --- CLI ---------------------------------------------------------------------


def test_main_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    document = tmp_path / "doc.txt"
    document.write_text(DOC, encoding="utf-8")
    queries = _write_queries(
        tmp_path / "q.json",
        [
            {"id": "q1", "language": "en", "query": "a", "expected_answer_spans": ["alpha"]},
            {"id": "q2", "language": "de", "query": "b", "expected_answer_spans": ["delta"]},
        ],
    )
    fake = FakeSearch({})
    monkeypatch.setattr(pipeline, "search_chunks", fake)
    monkeypatch.setattr(evaluation, "_git_sha", lambda: "abc1234")
    out = tmp_path / "out"

    exit_code = evaluation.main(
        [
            "--document", str(document),
            "--queries", str(queries),
            "--chunk-size", "2",
            "--overlap", "0",
            "--candidate-k", "4",
            "--output-dir", str(out),
            "--name", "run",
            "--sweep",
        ]
    )  # fmt: skip

    assert exit_code == 0
    report = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert set(report) == {"metadata", "metrics", "per_query", "sweep"}
    meta = report["metadata"]
    assert meta["git_sha"] == "abc1234"
    assert meta["chunk_size"] == 2
    assert meta["overlap"] == 0
    assert meta["num_chunks"] == 3
    assert meta["ks"] == [1, 3, 5]
    assert meta["candidate_k"] == 4
    assert meta["embedding_model"]
    assert meta["reranker_model"]
    assert meta["timestamp_utc"].endswith("+00:00")
    assert set(report["metrics"]) == {"dense", "dense+rerank"}
    dense = report["metrics"]["dense"]
    assert set(dense) >= {"hit_at_k", "recall_at_k", "mrr", "mrr_cutoff", "unreachable_queries"}
    assert set(dense["hit_at_k"]) == {"1", "3", "5"}
    assert [r["id"] for r in report["per_query"]["dense+rerank"]] == ["q1", "q2"]
    # 4 chunk sizes x 2 overlaps; none of the combos is invalid.
    assert len(report["sweep"]) == 8
    assert {(s["chunk_size"], s["overlap"]) for s in report["sweep"]} >= {(8, 4), (50, 0)}

    rerank_flags = {c["use_reranker"] for c in fake.calls}
    assert rerank_flags == {True, False}
    assert {c["candidate_k"] for c in fake.calls} == {4}

    markdown = (out / "run.md").read_text(encoding="utf-8")
    assert "| dense |" in markdown
    assert "| dense+rerank |" in markdown
    assert "hit@1" in markdown and "recall@5" in markdown and "MRR@5" in markdown
    assert "Unreachable queries" in markdown
    assert "Chunking sweep" in markdown
    assert "Wrote" in capsys.readouterr().out


@pytest.mark.slow
def test_evaluate_end_to_end_with_real_models() -> None:
    document_text = DOCUMENT_PATH.read_text(encoding="utf-8")
    queries = load_queries(QUERIES_PATH)
    for use_reranker in (False, True):
        metrics, records = evaluate(
            document_text,
            queries,
            chunk_size=50,
            overlap=0,
            use_reranker=use_reranker,
            candidate_k=5,
        )
        assert metrics.num_chunks > 0
        assert len(records) == len(queries)
        assert 0.0 <= metrics.mrr <= 1.0
        for value in [*metrics.hit_at_k.values(), *metrics.recall_at_k.values()]:
            assert 0.0 <= value <= 1.0
