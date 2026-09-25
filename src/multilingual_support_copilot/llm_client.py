from ollama import chat
from multilingual_support_copilot.models import AnswerResponse

MODEL_NAME = "qwen3:4b"


def generate_answer(prompt: str) -> str:
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        
    )

    return response["message"]["content"].strip()

def generate_structured_answer(prompt: str) -> AnswerResponse:
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format=AnswerResponse.model_json_schema(),
    )

    return AnswerResponse.model_validate_json(
        response["message"]["content"]
    )