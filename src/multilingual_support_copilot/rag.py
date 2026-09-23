from pathlib import Path

from multilingual_support_copilot.llm_client import generate_answer
from multilingual_support_copilot.pipeline import search_document
from multilingual_support_copilot.prompt_builder import build_grounded_prompt


def answer_question(
    query: str,
    file_path: Path,
    chunk_size: int = 50,
    top_k: int = 3,
    overlap: int = 0,
) -> str:
    retrieved_chunks = search_document(
        query=query,
        file_path=file_path,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        
    )

    prompt = build_grounded_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    answer = generate_answer(prompt)

    return answer
