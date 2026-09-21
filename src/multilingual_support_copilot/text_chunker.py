def chunk_text(text:str, chunk_size: int = 50)->list[str]:
    words= text.split()

    if not words:
        return[]

    chunks =[]

    for i in range(0, len(words), chunk_size):
        chunk_words= words[i:i+chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks