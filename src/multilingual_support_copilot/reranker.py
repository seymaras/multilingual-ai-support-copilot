"""Lazily loaded multilingual cross-encoder reranker."""

from functools import lru_cache
from typing import TYPE_CHECKING

from multilingual_support_copilot.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def get_reranker_model() -> "CrossEncoder":
    from sentence_transformers import CrossEncoder

    model: CrossEncoder = CrossEncoder(get_settings().reranker_model_name)
    return model


def rerank_chunks(
    query: str,
    retrieved_chunks: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Re-score ``retrieved_chunks`` with the cross-encoder, highest score first."""
    if not retrieved_chunks:
        return []

    pairs = [(query, chunk) for chunk, _ in retrieved_chunks]
    scores = get_reranker_model().predict(pairs, show_progress_bar=False)

    reranked = [
        (chunk, float(score)) for (chunk, _), score in zip(retrieved_chunks, scores, strict=True)
    ]
    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked
