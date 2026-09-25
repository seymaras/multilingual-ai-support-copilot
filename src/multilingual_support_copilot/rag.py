from pathlib import Path

from multilingual_support_copilot.llm_client import (
    generate_answer,
    generate_structured_answer,
)
from multilingual_support_copilot.models import AnswerResponse
from multilingual_support_copilot.pipeline import search_document
from multilingual_support_copilot.prompt_builder import build_grounded_prompt


def answer_question(
    query: str,
    file_path: Path,
    chunk_size: int = 50,
    overlap: int = 0,
    top_k: int = 3,
    use_reranker: bool = False,
    candidate_k: int = 5,
) -> str:
    retrieved_chunks = search_document(
        query=query,
        file_path=file_path,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        use_reranker=use_reranker,
        candidate_k=candidate_k,
    )

    prompt = build_grounded_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    answer = generate_answer(prompt)

    return answer

def answer_question_structured(
    query: str,
    file_path: Path,
    chunk_size: int = 50,
    overlap: int = 0,
    top_k: int = 3,
    use_reranker: bool = False,
    candidate_k: int = 5,
) -> AnswerResponse:
    retrieved_chunks = search_document(
        query=query,
        file_path=file_path,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        use_reranker=use_reranker,
        candidate_k=candidate_k,
    )

    prompt = build_grounded_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    response = generate_structured_answer(prompt)

    return response