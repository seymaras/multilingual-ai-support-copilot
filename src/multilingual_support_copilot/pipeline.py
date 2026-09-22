from pathlib import Path

from multilingual_support_copilot.document_loader import load_text_file
from multilingual_support_copilot.text_chunker import chunk_text
from multilingual_support_copilot.retriever import retrieve_top_k_chunks


def search_document(
    query: str,
    file_path: Path,
    chunk_size: int = 50,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    text = load_text_file(file_path)

    chunks = chunk_text(text, chunk_size)

    results = retrieve_top_k_chunks(
        query,
        chunks,
        top_k,
    )

    return results