from sentence_transformers.util import cos_sim

from multilingual_support_copilot.embeddings import embed_texts


def retrieve_top_chunk(
    query: str,
    chunks: list[str]
) -> tuple[str, float]:

    query_embedding = embed_texts([query])[0]
    chunk_embeddings = embed_texts(chunks)

    best_chunk = ""
    best_score = float("-inf")

    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
        score = cos_sim(query_embedding, chunk_embedding).item()

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk, best_score

def retrieve_top_k_chunks(
    query: str,
    chunks: list[str],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    query_embedding = embed_texts([query])[0]
    chunk_embeddings = embed_texts(chunks)

    scored_chunks = []

    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
        score = cos_sim(query_embedding, chunk_embedding).item()
        scored_chunks.append((chunk, score))

    scored_chunks.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored_chunks[:top_k]