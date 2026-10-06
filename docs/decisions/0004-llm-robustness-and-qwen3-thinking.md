# ADR 0004: LLM error handling, structured-output retries and qwen3 thinking

- **Status:** Accepted (Stage 0)
- **Date:** 2026-10-06

## Context
- The Ollama calls had no error handling. If the server was down, slow, or the model was not pulled,
  the caller saw a raw `ConnectionError` or `httpx.ReadTimeout`. The Ollama client does **not** wrap
  timeouts itself.
- Invalid structured output raised an uncaught `pydantic.ValidationError`.
- `qwen3:4b` is a hybrid reasoning model. With thinking left on, every answer gets slower and costs
  more tokens, and `<think>…</think>` text can leak into the content the customer sees.

## Decision
- **Error taxonomy** (`errors.py`), all subclasses of `CopilotError`:
  | Error | Raised when |
  |---|---|
  | `LLMUnavailableError` | connection refused, timeout (with the configured seconds), HTTP 404 (with an `ollama pull <model>` hint), other HTTP errors |
  | `LLMResponseError` | the output still fails the schema after all retries; it carries `attempts` and `last_error` |
- **Retries only for invalid output.** `generate_structured_answer` tries up to
  `1 + COPILOT_LLM_MAX_RETRIES` times (default 3 attempts) when validation fails, and logs a warning
  each time. Availability errors are **not** retried, because waiting longer on a server that is down
  only adds latency. Worst-case latency is `(1 + retries) × timeout`.
- **Thinking off, plus a safety net.**
  - Every request sends `think=COPILOT_LLM_THINK`, which defaults to `false`.
  - `strip_think_blocks()` also removes:
    - closed `<think>` blocks;
    - a leading orphan `…</think>`;
    - a trailing unclosed `<think>…`.
  - This covers model templates that emit empty think tags even with thinking disabled.

## Alternatives considered
| Option | Why not |
|---|---|
| Use a non-reasoning model | Still an option, since the model name is configurable. qwen3 has good TR/DE quality for its size. |
| Retry on timeouts too | It doubles the worst-case latency for a support widget. Circuit breaking and failover belong to the provider abstraction in Stage 2 or 3. |
| Repair invalid JSON with a parser | `format=<schema>` already constrains decoding, so invalid output is rare. Re-asking is simpler and the cases are visible in the logs. |

## Trade-offs
- The "no context" answer is currently English. Localising it needs language detection (Stage 2 or 3).
- The structured answer's `sources` and `has_evidence` fields are still produced by the LLM. Deriving
  them deterministically from retrieval is Stage 3 scope.
