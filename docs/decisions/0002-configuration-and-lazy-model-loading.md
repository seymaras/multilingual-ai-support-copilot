# ADR 0002: Typed configuration and lazy model loading

- **Status:** Accepted (Stage 0)
- **Date:** 2026-10-06

## Context
- Model names, the Ollama host and the retrieval parameters were hard-coded in several modules.
- Retrieval parameters (`top_k`, `candidate_k`, `chunk_size`) had no bounds, so a client could ask for
  `top_k=10**9` or `candidate_k < top_k`.
- Both ML models were loaded at **import time**. Importing anything loaded torch plus about 500 MB of
  weights, which made tests slow and tied them to the real models. `/health` could not start without
  the models either.

## Decision
- **`pydantic-settings`.** A single `Settings` class (`config.py`) reads `COPILOT_*` variables and an
  optional `.env` file, and is documented in `.env.example`.
  - A cross-field validator enforces:
    - `chunk_overlap < chunk_size <= max_chunk_size`
    - `default_top_k <= default_candidate_k`
    - every default stays within its maximum
  - `get_settings()` is cached.
- **Request bounds come from settings.** `QuestionRequest` fills `top_k` and `candidate_k` from the
  settings defaults and rejects values outside `[1, max]`, `candidate_k < top_k`, and over-long queries.
- **Lazy singletons.** `get_embedding_model()`, `get_reranker_model()` and `get_ollama_client()` are
  `lru_cache(maxsize=1)` getters.
  - `sentence_transformers` (and so torch) is imported inside the getter only.
  - The retriever computes cosine similarity as a dot product of normalised NumPy vectors instead of
    importing `sentence_transformers.util`.
  - A subprocess test asserts that importing the API, RAG, pipeline and eval modules does not import
    torch.
- **Test isolation.** An autouse fixture:
  - replaces the real getters with functions that fail in any test not marked `slow`;
  - clears all caches;
  - removes `COPILOT_*` variables and runs each test from an empty working directory, so a developer's
    `.env` cannot change the results.
  Real-model tests carry `@pytest.mark.slow` and are excluded by default (`pytest -m slow` runs them).

## Alternatives considered
| Option | Why not |
|---|---|
| A dependency-injection container | Too heavy for the current size. The getters are easy to replace with one later. |
| Load the models in the FastAPI lifespan | It ties loading to the web server and gives no benefit to the CLI and eval. We may still add a lifespan *warm-up* in Stage 2 so the first request is not slow. |
| Keep import-time loading and mock `sentence_transformers` in tests | Imports stay slow and fragile, and the bug simply moves elsewhere. |

## Trade-offs
- The first request after start-up pays the model-load cost (a few seconds). A warm-up hook is planned
  for Stage 2.
- Request validation now depends on global settings. The tests clear the cache explicitly.
