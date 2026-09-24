from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

reranker = CrossEncoder(RERANKER_MODEL_NAME)


def rerank_chunks(
    query: str,
    retrieved_chunks: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    pairs = [
        (query, chunk)
        for chunk, _ in retrieved_chunks
    ]

    scores = reranker.predict(pairs)

    reranked_results = [
        (chunk, float(score))
        for (chunk, _), score in zip(retrieved_chunks, scores)
    ]

    reranked_results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return reranked_results