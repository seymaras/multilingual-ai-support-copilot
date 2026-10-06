"""Lazily loaded multilingual bi-encoder.

``sentence_transformers`` (and torch) are imported on first use only, so importing
this package stays fast and tests never load real models by accident.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from multilingual_support_copilot.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model_name)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return an ``(len(texts), dim)`` matrix of L2-normalized embeddings."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)
