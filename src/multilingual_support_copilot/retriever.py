"""Dense retrieval over in-memory chunks using the shared bi-encoder."""

from multilingual_support_copilot import embeddings


def retrieve_top_k_chunks(
    query: str,
    chunks: list[str],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Return up to ``top_k`` ``(chunk, cosine_score)`` pairs, highest score first.

    Query and chunks are embedded in a single ``encode`` call. Ties keep the
    original chunk order (stable sort).
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not chunks:
        return []

    vectors = embeddings.embed_texts([query, *chunks])
    query_vector, chunk_vectors = vectors[0], vectors[1:]
    # Rows are L2-normalized, so the dot product is the cosine similarity.
    scores = chunk_vectors @ query_vector

    order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
    return [(chunks[i], float(scores[i])) for i in order[:top_k]]
