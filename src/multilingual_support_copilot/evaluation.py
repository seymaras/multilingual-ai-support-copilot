"""Retrieval evaluation over the production chunker and ``pipeline.search_chunks``.

Gold labels are short answer *spans*; a chunk is relevant to a query when it contains
one of the query's spans (after whitespace normalization and case folding). Metrics:

- ``hit@k``: fraction of queries with at least one span contained in some top-k chunk.
- ``recall@k``: per query, fraction of its spans covered by the top-k chunks; averaged.
- ``MRR@K`` (K = max(ks)): mean of 1/rank of the first chunk containing any span, else 0.

A query is *unreachable* when some span is not wholly contained in any chunk (for
example because it straddles a chunk boundary); such queries can never reach full recall.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multilingual_support_copilot import pipeline
from multilingual_support_copilot.config import get_settings
from multilingual_support_copilot.text_chunker import chunk_text

CONFIGS: dict[str, bool] = {"dense": False, "dense+rerank": True}
SWEEP_CHUNK_SIZES = (8, 16, 32, 50)
SWEEP_OVERLAPS = (0, 4)
PREVIEW_CHARS = 80


@dataclass(frozen=True)
class EvalQuery:
    id: str
    language: str
    query: str
    expected_answer_spans: tuple[str, ...]


@dataclass
class RetrievalMetrics:
    hit_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    mrr_cutoff: int
    num_queries: int
    num_chunks: int
    unreachable_queries: list[str]


def normalize(text: str) -> str:
    """Collapse whitespace (as the chunker does) and case-fold."""
    return " ".join(text.split()).casefold()


def chunk_contains_span(chunk: str, span: str) -> bool:
    return normalize(span) in normalize(chunk)


def load_queries(path: Path) -> list[EvalQuery]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of queries")

    queries: list[EvalQuery] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: entry {index} is not an object")
        query_id = str(item.get("id", ""))
        if not query_id:
            raise ValueError(f"{path}: entry {index} has no id")
        if query_id in seen:
            raise ValueError(f"{path}: duplicate query id {query_id!r}")
        seen.add(query_id)

        spans = item.get("expected_answer_spans")
        if (
            not isinstance(spans, list)
            or not spans
            or not all(isinstance(s, str) and s.strip() for s in spans)
        ):
            raise ValueError(
                f"{path}: query {query_id!r} needs a non-empty list of "
                "non-empty strings in 'expected_answer_spans'"
            )
        queries.append(
            EvalQuery(
                id=query_id,
                language=str(item.get("language", "")),
                query=str(item["query"]),
                expected_answer_spans=tuple(spans),
            )
        )
    return queries


def validate_spans(document_text: str, queries: Iterable[EvalQuery]) -> None:
    """Raise ``ValueError`` naming every span that does not occur in the document."""
    document = normalize(document_text)
    missing = [
        f"{q.id}: {span!r}"
        for q in queries
        for span in q.expected_answer_spans
        if normalize(span) not in document
    ]
    if missing:
        raise ValueError("Spans not found in document:\n  " + "\n  ".join(missing))


def first_relevant_rank(retrieved: Sequence[str], spans: Sequence[str]) -> int | None:
    """1-based rank of the first chunk containing any span, or ``None``."""
    for rank, chunk in enumerate(retrieved, start=1):
        if any(chunk_contains_span(chunk, span) for span in spans):
            return rank
    return None


def _span_recall(retrieved: Sequence[str], spans: Sequence[str]) -> float:
    covered = sum(any(chunk_contains_span(c, span) for c in retrieved) for span in spans)
    return covered / len(spans)


def _rerank_pool(top_k: int, candidate_k: int | None) -> int:
    """Candidate pool size ``pipeline.search_chunks`` reranks for a given ``top_k``."""
    return max(candidate_k or top_k, top_k)


def _top_k_as_in_production(
    query: str,
    chunks: list[str],
    retrieved_max_k: list[str],
    *,
    k: int,
    max_k: int,
    use_reranker: bool,
    candidate_k: int | None,
) -> list[str]:
    """Top-k exactly as production returns it for ``top_k=k``.

    Dense rankings are prefix-consistent, so slicing the max-k result is exact. With
    reranking, production reranks a pool of ``max(candidate_k, k)``; if that differs from
    the pool used for the max-k call, re-run with ``top_k=k``.
    """
    if not use_reranker or _rerank_pool(k, candidate_k) == _rerank_pool(max_k, candidate_k):
        return retrieved_max_k[:k]
    results = pipeline.search_chunks(
        query, chunks, top_k=k, use_reranker=True, candidate_k=candidate_k
    )
    return [chunk for chunk, _ in results]


def _preview(chunk: str) -> str:
    return chunk if len(chunk) <= PREVIEW_CHARS else chunk[: PREVIEW_CHARS - 1] + "…"


def evaluate(
    document_text: str,
    queries: Sequence[EvalQuery],
    *,
    chunk_size: int,
    overlap: int,
    ks: Sequence[int] = (1, 3, 5),
    use_reranker: bool = False,
    candidate_k: int | None = None,
) -> tuple[RetrievalMetrics, list[dict[str, Any]]]:
    if not queries:
        raise ValueError("No queries to evaluate")
    if not ks or min(ks) < 1:
        raise ValueError("ks must be a non-empty sequence of positive integers")
    validate_spans(document_text, queries)

    ks = sorted(set(ks))
    max_k = max(ks)
    chunks = chunk_text(document_text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("Document produced no chunks")

    hits = dict.fromkeys(ks, 0.0)
    recalls = dict.fromkeys(ks, 0.0)
    reciprocal_ranks = 0.0
    unreachable: list[str] = []
    records: list[dict[str, Any]] = []

    for q in queries:
        spans = q.expected_answer_spans
        if not all(any(chunk_contains_span(c, s) for c in chunks) for s in spans):
            unreachable.append(q.id)

        results = pipeline.search_chunks(
            q.query,
            chunks,
            top_k=max_k,
            use_reranker=use_reranker,
            candidate_k=candidate_k,
        )
        retrieved = [chunk for chunk, _ in results]
        rank = first_relevant_rank(retrieved, spans)

        for k in ks:
            top = _top_k_as_in_production(
                q.query,
                chunks,
                retrieved,
                k=k,
                max_k=max_k,
                use_reranker=use_reranker,
                candidate_k=candidate_k,
            )
            if first_relevant_rank(top, spans) is not None:
                hits[k] += 1
            recalls[k] += _span_recall(top, spans)
        if rank is not None:
            reciprocal_ranks += 1.0 / rank

        records.append(
            {
                "id": q.id,
                "language": q.language,
                "query": q.query,
                "expected_answer_spans": list(spans),
                "first_relevant_rank": rank,
                "retrieved": [
                    {"rank": i, "score": float(score), "preview": _preview(chunk)}
                    for i, (chunk, score) in enumerate(results, start=1)
                ],
            }
        )

    n = len(queries)
    metrics = RetrievalMetrics(
        hit_at_k={k: hits[k] / n for k in ks},
        recall_at_k={k: recalls[k] / n for k in ks},
        mrr=reciprocal_ranks / n,
        mrr_cutoff=max_k,
        num_queries=n,
        num_chunks=len(chunks),
        unreachable_queries=unreachable,
    )
    return metrics, records


# --- CLI ---------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _metrics_json(metrics: RetrievalMetrics) -> dict[str, Any]:
    data = asdict(metrics)
    data["hit_at_k"] = {str(k): v for k, v in metrics.hit_at_k.items()}
    data["recall_at_k"] = {str(k): v for k, v in metrics.recall_at_k.items()}
    return data


def _render_markdown(report: dict[str, Any], results: dict[str, RetrievalMetrics]) -> str:
    meta = report["metadata"]
    ks: list[int] = meta["ks"]
    header = (
        ["config"] + [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] + [f"MRR@{max(ks)}"]
    )
    lines = [
        f"# Retrieval evaluation: {meta['name']}",
        "",
        f"- Generated: {meta['timestamp_utc']} (git `{meta['git_sha'] or 'unknown'}`)",
        f"- Document: `{meta['document']}`; queries: `{meta['queries']}` "
        f"({meta['num_queries']} queries)",
        f"- Chunking: chunk_size={meta['chunk_size']}, overlap={meta['overlap']} "
        f"-> {meta['num_chunks']} chunks",
        f"- Embedding model: `{meta['embedding_model']}`; reranker: `{meta['reranker_model']}` "
        f"(candidate_k={meta['candidate_k']})",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for name, m in results.items():
        row = (
            [name]
            + [f"{m.hit_at_k[k]:.3f}" for k in ks]
            + [f"{m.recall_at_k[k]:.3f}" for k in ks]
            + [f"{m.mrr:.3f}"]
        )
        lines.append("| " + " | ".join(row) + " |")

    unreachable = next(iter(results.values())).unreachable_queries
    lines += [
        "",
        "Unreachable queries (a span is not wholly inside any chunk): "
        + (", ".join(unreachable) if unreachable else "none"),
    ]
    if meta["num_chunks"] <= max(ks):
        lines.append(
            f"Note: only {meta['num_chunks']} chunks exist, so cut-offs >= "
            f"{meta['num_chunks']} retrieve the whole document."
        )

    sweep = report.get("sweep")
    if sweep:
        k_mid = ks[len(ks) // 2]
        lines += [
            "",
            "## Chunking sweep (context only)",
            "",
            f"| chunk_size | overlap | chunks | config | hit@{ks[0]} | hit@{k_mid} | "
            f"recall@{k_mid} | MRR | unreachable |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        first_k = str(ks[0])
        for entry in sweep:
            for name, m in entry["configs"].items():
                lines.append(
                    f"| {entry['chunk_size']} | {entry['overlap']} | {entry['num_chunks']} | "
                    f"{name} | {m['hit_at_k'][first_k]:.3f} | {m['hit_at_k'][str(k_mid)]:.3f} | "
                    f"{m['recall_at_k'][str(k_mid)]:.3f} | {m['mrr']:.3f} | "
                    f"{len(m['unreachable_queries'])} |"
                )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="copilot-eval",
        description="Evaluate retrieval (dense and dense+rerank) on a labelled query set.",
    )
    parser.add_argument("--document", type=Path, default=Path("data/evaluation_document.txt"))
    parser.add_argument("--queries", type=Path, default=Path("data/evaluation_queries.json"))
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--candidate-k", type=int, default=settings.default_candidate_k)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output-dir", type=Path, default=Path("docs/evals"))
    parser.add_argument("--name", default="baseline")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Also evaluate chunk_size in (8, 16, 32, 50) x overlap in (0, 4).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    ks = sorted(set(args.ks))
    if min(ks) < 1:
        raise SystemExit("--ks values must be positive")

    document_text = args.document.read_text(encoding="utf-8")
    queries = load_queries(args.queries)

    results: dict[str, RetrievalMetrics] = {}
    per_query: dict[str, list[dict[str, Any]]] = {}
    for name, use_reranker in CONFIGS.items():
        metrics, records = evaluate(
            document_text,
            queries,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            ks=ks,
            use_reranker=use_reranker,
            candidate_k=args.candidate_k,
        )
        results[name] = metrics
        per_query[name] = records

    sweep: list[dict[str, Any]] = []
    if args.sweep:
        for size in SWEEP_CHUNK_SIZES:
            for overlap in SWEEP_OVERLAPS:
                if overlap >= size:
                    continue
                configs: dict[str, Any] = {}
                for name, use_reranker in CONFIGS.items():
                    m, _ = evaluate(
                        document_text,
                        queries,
                        chunk_size=size,
                        overlap=overlap,
                        ks=ks,
                        use_reranker=use_reranker,
                        candidate_k=args.candidate_k,
                    )
                    configs[name] = _metrics_json(m)
                sweep.append(
                    {
                        "chunk_size": size,
                        "overlap": overlap,
                        "num_chunks": m.num_chunks,
                        "configs": configs,
                    }
                )

    first = next(iter(results.values()))
    report: dict[str, Any] = {
        "metadata": {
            "name": args.name,
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "document": str(args.document),
            "queries": str(args.queries),
            "num_queries": first.num_queries,
            "embedding_model": settings.embedding_model_name,
            "reranker_model": settings.reranker_model_name,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "num_chunks": first.num_chunks,
            "ks": ks,
            "candidate_k": args.candidate_k,
        },
        "metrics": {name: _metrics_json(m) for name, m in results.items()},
        "per_query": per_query,
        "sweep": sweep,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.name}.json"
    md_path = args.output_dir / f"{args.name}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report, results), encoding="utf-8")

    print(
        f"{first.num_queries} queries, {first.num_chunks} chunks "
        f"(chunk_size={args.chunk_size}, overlap={args.overlap})"
    )
    for name, m in results.items():
        hits = " ".join(f"hit@{k}={v:.3f}" for k, v in m.hit_at_k.items())
        recalls = " ".join(f"recall@{k}={v:.3f}" for k, v in m.recall_at_k.items())
        print(f"  {name:<13} {hits} {recalls} MRR@{m.mrr_cutoff}={m.mrr:.3f}")
    if first.unreachable_queries:
        print(f"  unreachable: {', '.join(first.unreachable_queries)}")
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
