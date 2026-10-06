"""Retrieval-augmented answering over server-managed documents."""

from multilingual_support_copilot.llm_client import generate_answer, generate_structured_answer
from multilingual_support_copilot.models import AnswerResponse, QuestionRequest
from multilingual_support_copilot.pipeline import search_document
from multilingual_support_copilot.prompt_builder import build_grounded_prompt

# Returned without calling the LLM when retrieval finds nothing.
# Localization follows once language detection lands (Stage 2/3).
NO_CONTEXT_ANSWER = "I don't know. I could not find relevant information in the documents."


def _retrieve(
    query: str,
    document_id: str,
    *,
    top_k: int | None,
    use_reranker: bool,
    candidate_k: int | None,
) -> list[tuple[str, float]]:
    return search_document(
        query=query,
        document_id=document_id,
        top_k=top_k,
        use_reranker=use_reranker,
        candidate_k=candidate_k,
    )


def answer_question(
    query: str,
    document_id: str,
    *,
    top_k: int | None = None,
    use_reranker: bool = False,
    candidate_k: int | None = None,
) -> str:
    chunks = _retrieve(
        query, document_id, top_k=top_k, use_reranker=use_reranker, candidate_k=candidate_k
    )
    if not chunks:
        return NO_CONTEXT_ANSWER
    return generate_answer(build_grounded_prompt(query=query, retrieved_chunks=chunks))


def answer_question_structured(
    query: str,
    document_id: str,
    *,
    top_k: int | None = None,
    use_reranker: bool = False,
    candidate_k: int | None = None,
) -> AnswerResponse:
    chunks = _retrieve(
        query, document_id, top_k=top_k, use_reranker=use_reranker, candidate_k=candidate_k
    )
    if not chunks:
        return AnswerResponse(answer=NO_CONTEXT_ANSWER, has_evidence=False, sources=[])
    return generate_structured_answer(build_grounded_prompt(query=query, retrieved_chunks=chunks))


def answer_request(request: QuestionRequest) -> AnswerResponse:
    """Adapter from the validated request model (used by the API in Stage 2)."""
    return answer_question_structured(
        request.query,
        request.document_id,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
        candidate_k=request.candidate_k,
    )
