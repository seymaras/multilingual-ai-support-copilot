def build_grounded_prompt(
    query: str,
    retrieved_chunks: list[tuple[str, float]],
) -> str:
    context = "\n\n".join(
        chunk for chunk, _ in retrieved_chunks
    )

    prompt = (
        "Answer the user's question using only the provided context.\n"
        "If the context does not contain enough information, say that you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

    return prompt
    