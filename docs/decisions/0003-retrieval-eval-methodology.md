# ADR 0003: Retrieval evaluation with answer spans and the production chunker

- **Status:** Accepted (Stage 0)
- **Date:** 2026-10-06

## Context
The previous eval reported "hit@1 0.8 → 0.9 with reranking", but it measured a different system than
the one in production:

| | Old eval | Production |
|---|---|---|
| Chunking | one line = one chunk | 50-word sliding window |
| Gold label | exact chunk text | n/a |
| Metric name | "recall" | n/a |
| Retrieval code | its own copy | `pipeline.py` |

Two further problems:
- Matching on exact chunk text breaks as soon as chunking changes, which makes chunking itself
  impossible to evaluate.
- The number it called "recall" was actually hit-rate@k.

## Decision
- **Gold labels are answer spans.** Each query lists the passages a correct context must contain
  (`expected_answer_spans`). A retrieved chunk counts as relevant if it contains a span, after
  whitespace normalisation and casefolding.
  - Spans must be distinctive. "14 days" appears in two facts, so the label is
    "return period is 14 days".
  - `validate_spans` fails fast if a label no longer appears in the document.
- **Same code path as production.** The eval uses the production `chunk_text` and
  `pipeline.search_chunks`, with chunk parameters taken from `Settings` by default.
- **Metrics, defined precisely:**
  - **hit@k:** the fraction of queries where some top-k chunk contains at least one span.
  - **recall@k:** for each query, the fraction of its spans covered by the top-k chunks, averaged over
    queries. It differs from hit@k only for multi-span queries (q11, q12).
  - **MRR@K:** the mean of 1/rank of the first relevant chunk, counted as 0 beyond K = max(ks) = 5.
  - **Unreachable queries:** a span that no chunk contains whole, because it straddles a chunk
    boundary. These expose chunking problems instead of hiding them.
- **Output.** `copilot-eval` writes `docs/evals/<name>.json`, with metadata, git SHA, metrics and
  per-query ranks, plus a Markdown table. `--sweep` adds a chunk_size × overlap grid for context.

## Baseline (Stage 0, git `b1ffd46`)
14 queries (TR/DE/EN), chunk_size=50, overlap=0, giving 3 chunks:

| config | hit@1 | hit@3 | recall@3 | MRR@5 |
|---|---|---|---|---|
| dense | 0.571 | 1.000 | 0.964 | 0.774 |
| dense + rerank | **0.929** | 1.000 | 0.964 | **0.964** |

The numbers are **not comparable** with the old 0.8 / 0.9: the chunking, the labels and the query set
are all different. This is a methodology reset, not a regression.

## Known limitations
- **The corpus is tiny.** The document has 107 words, so 50-word chunks give only 3 chunks, and any
  cut-off of 3 or more returns the whole document. At this size only hit@1 and MRR carry signal.
- **q12 is unreachable at 50/0.** Its "premium support" span is split across two chunks. The sweep
  shows that a small overlap removes most of these misses. 32/4 with rerank scores 1.0 on every
  metric, but on 14 queries that is a hint, not a result.
- Expanding the corpus and query set and choosing chunk settings on held-out data are Stage 4 work
  ("Eval 2.0").

## Alternatives considered
| Option | Why not |
|---|---|
| Label by chunk ID | IDs change whenever chunking changes, so chunking could not be tuned. |
| Rate chunks with an LLM judge | Not deterministic and costs money on every run. It may later complement the span labels for answer quality, not replace them. |
