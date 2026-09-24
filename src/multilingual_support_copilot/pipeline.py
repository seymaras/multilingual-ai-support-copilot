from pathlib import Path

from multilingual_support_copilot.document_loader import load_text_file
from multilingual_support_copilot.text_chunker import chunk_text
from multilingual_support_copilot.retriever import retrieve_top_k_chunks
from multilingual_support_copilot.reranker import rerank_chunks

def search_document(
    query: str,
    file_path: Path,
    chunk_size: int = 50,
    overlap: int = 0,
    top_k: int = 3,
    use_reranker: bool = False,
    candidate_k: int = 5,
) -> list[tuple[str, float]]:
    text = load_text_file(file_path)

    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not use_reranker:
        return retrieve_top_k_chunks(
            query,
            chunks,
            top_k,
        )

    candidates = retrieve_top_k_chunks(
        query,
        chunks,
        candidate_k,
    )

    reranked_results = rerank_chunks(
        query=query,
        retrieved_chunks=candidates,
    )

    return reranked_results[:top_k]