"""Retrieval pipeline: dense retrieval, optionally followed by cross-encoder reranking."""

from multilingual_support_copilot.config import Settings, get_settings
from multilingual_support_copilot.document_loader import load_text_file
from multilingual_support_copilot.documents import DocumentRegistry
from multilingual_support_copilot.errors import EmptyDocumentError
from multilingual_support_copilot.reranker import rerank_chunks
from multilingual_support_copilot.retriever import retrieve_top_k_chunks
from multilingual_support_copilot.text_chunker import chunk_text


def search_chunks(
    query: str,
    chunks: list[str],
    *,
    top_k: int,
    use_reranker: bool = False,
    candidate_k: int | None = None,
) -> list[tuple[str, float]]:
    """Rank ``chunks`` for ``query``. Shared by the production path and the eval."""
    if not use_reranker:
        return retrieve_top_k_chunks(query, chunks, top_k)

    candidates = retrieve_top_k_chunks(query, chunks, max(candidate_k or top_k, top_k))
    return rerank_chunks(query=query, retrieved_chunks=candidates)[:top_k]


def search_document(
    query: str,
    document_id: str,
    *,
    top_k: int | None = None,
    use_reranker: bool = False,
    candidate_k: int | None = None,
    settings: Settings | None = None,
) -> list[tuple[str, float]]:
    settings = settings or get_settings()
    if top_k is None:
        top_k = settings.default_top_k
    if candidate_k is None:
        candidate_k = max(settings.default_candidate_k, top_k)
    if not 1 <= top_k <= settings.max_top_k:
        raise ValueError(f"top_k must be between 1 and {settings.max_top_k}")
    if not top_k <= candidate_k <= settings.max_candidate_k:
        raise ValueError(f"candidate_k must be between top_k and {settings.max_candidate_k}")

    path = DocumentRegistry(settings.documents_dir).resolve(document_id)
    text = load_text_file(path)
    chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        raise EmptyDocumentError(f"Document {document_id!r} contains no text")

    return search_chunks(
        query,
        chunks,
        top_k=top_k,
        use_reranker=use_reranker,
        candidate_k=candidate_k,
    )
