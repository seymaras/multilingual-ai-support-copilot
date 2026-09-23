def chunk_text(
    text: str,
    chunk_size: int = 50,
    overlap: int = 0,
) -> list[str]:
    words = text.split()

    if not words:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be greater than or equal to 0 "
            "and smaller than chunk_size."
        )

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]

        chunks.append(" ".join(chunk_words))

        if i + chunk_size >= len(words):
            break

    return chunks