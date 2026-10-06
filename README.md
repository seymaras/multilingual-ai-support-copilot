# Multilingual AI Support Copilot

A retrieval-augmented support assistant that answers customer questions in **Turkish, German and
English** from a company's own documents, and says "I don't know" when the documents do not cover
the question. It is built for EU hosting or self-hosting with [Ollama](https://ollama.com).

> Status: pre-pilot. Stage 0 (foundation hardening) is complete. See [Roadmap](#roadmap).

## How it works

```
question ──► dense retrieval ──► cross-encoder rerank ──► grounded prompt ──► LLM (Ollama) ──► AnswerResponse
             (multilingual         (multilingual            "answer only from      qwen3:4b,        {answer,
              MiniLM bi-encoder)    mMiniLM cross-encoder)   the context"           thinking off     has_evidence,
                                                                                                     sources}
```

- **Cross-lingual retrieval.** A Turkish question finds an English or German passage.
  - Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`
  - Reranker: `mmarco-mMiniLMv2-L12-H384-v1`
- **Structured output.** Ollama's JSON-schema `format=` produces an `AnswerResponse`. Output that fails
  the schema is retried, then reported as a clear error.
- **No context, no LLM call.** If retrieval returns nothing, the copilot answers "I don't know" with
  `has_evidence=false`.
- **Documents are referenced by server-managed ID**, never by file path
  ([ADR 0001](docs/decisions/0001-server-managed-document-ids.md)).

## Quickstart

Requires Python 3.11+ and Ollama.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ollama pull qwen3:4b
cp .env.example .env              # optional; every setting has a default
```

Put documents in `COPILOT_DOCUMENTS_DIR` (default `data/`) as `<id>.txt`. The file stem is the
document ID and must match `^[a-z0-9][a-z0-9_-]{0,63}$`.

```python
from multilingual_support_copilot.rag import answer_question_structured

answer_question_structured("İade için kaç günüm var?", "evaluation_document", use_reranker=True)
# AnswerResponse(answer='İade için 14 gününüz var.', has_evidence=True, sources=[...])
```

Run the API (only `/health` for now; the chat endpoints arrive in Stage 2):

```bash
uvicorn multilingual_support_copilot.api:app --reload
```

## Configuration

All settings are environment variables with the `COPILOT_` prefix, or entries in `.env`. The full list
is in [`.env.example`](.env.example). Bounds are validated at start-up
([ADR 0002](docs/decisions/0002-configuration-and-lazy-model-loading.md)).

| Variable | Default | Notes |
|---|---|---|
| `COPILOT_LLM_MODEL_NAME` | `qwen3:4b` | any Ollama model tag |
| `COPILOT_OLLAMA_HOST` | `http://localhost:11434` | |
| `COPILOT_OLLAMA_TIMEOUT_SECONDS` | `60` | per request |
| `COPILOT_LLM_MAX_RETRIES` | `2` | retries for schema-invalid output only |
| `COPILOT_LLM_THINK` | `false` | qwen3 reasoning mode ([ADR 0004](docs/decisions/0004-llm-robustness-and-qwen3-thinking.md)) |
| `COPILOT_DOCUMENTS_DIR` | `data` | |
| `COPILOT_CHUNK_SIZE` / `COPILOT_CHUNK_OVERLAP` | `50` / `0` | words; server-side only |
| `COPILOT_DEFAULT_TOP_K` / `COPILOT_MAX_TOP_K` | `3` / `20` | |
| `COPILOT_DEFAULT_CANDIDATE_K` / `COPILOT_MAX_CANDIDATE_K` | `5` / `50` | must be ≥ `top_k` |

## Development

```bash
pytest                    # fast suite (~1 s): no model downloads, no Ollama
pytest -m slow            # real models + live Ollama (skipped if Ollama is unreachable)
ruff check . && ruff format --check .
mypy                      # strict
```

- Models load lazily on first use, so importing the package never imports torch.
- The fast suite fails any test that tries to load a real model.

## Evaluation

```bash
copilot-eval --output-dir docs/evals --name my-run --sweep
```

- Queries carry **expected answer spans**. Retrieval runs through the production chunker and pipeline,
  and the report covers hit@k, recall@k, MRR and spans that straddle chunk boundaries
  ([ADR 0003](docs/decisions/0003-retrieval-eval-methodology.md)).
- Stage 0 baseline ([full report](docs/evals/baseline-stage0.md)): 14 TR/DE/EN queries,
  chunk_size=50, overlap=0.

| config | hit@1 | hit@3 | recall@3 | MRR@5 |
|---|---|---|---|---|
| dense | 0.571 | 1.000 | 0.964 | 0.774 |
| dense + rerank | **0.929** | 1.000 | 0.964 | **0.964** |

The eval corpus is tiny (3 chunks), so read these numbers as a regression guard, not as a quality
claim. Stage 4 expands the corpus and query set.

## Architecture decisions

- [0001: Reference documents by server-managed IDs](docs/decisions/0001-server-managed-document-ids.md)
- [0002: Typed configuration and lazy model loading](docs/decisions/0002-configuration-and-lazy-model-loading.md)
- [0003: Retrieval evaluation with answer spans](docs/decisions/0003-retrieval-eval-methodology.md)
- [0004: LLM error handling and qwen3 thinking](docs/decisions/0004-llm-robustness-and-qwen3-thinking.md)

## Known limitations

- Single flat `.txt` documents only. Each query re-reads, re-chunks and re-embeds the document
  (persistence and ingestion arrive in Stage 1).
- `sources` and `has_evidence` are still produced by the LLM rather than derived from retrieval
  (Stage 3). In a manual run the answer to "İade için kaç günüm var?" was correct (14 days), but the
  cited source was the wrong sentence ("…14 days to verify their email address").
- The "I don't know" fallback text is English only.
- No HTTP chat endpoint, auth or multi-tenancy yet (Stage 2).

## Roadmap

| Stage | Scope |
|---|---|
| 0 ✅ | Foundation: security fix, config, lazy models, error handling, honest eval |
| 1 | Ingestion and persistence (PostgreSQL + pgvector) |
| 2 | API, multi-tenancy, streaming |
| 3 | Guardrails: deterministic citations, hand-off to a human |
| 4 | Eval 2.0 and observability |
| 5 | Admin panel, embeddable widget, deploy in Germany |
