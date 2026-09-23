from ollama import chat


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