from multilingual_support_copilot.prompt_builder import build_grounded_prompt


def test_build_grounded_prompt_contains_context_and_question() -> None:
    retrieved_chunks = [
        ("The return period is 14 days.", 0.82),
        ("Returned products must be unused.", 0.67),
    ]

    query = "İade için kaç günüm var?"

    prompt = build_grounded_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    assert "The return period is 14 days." in prompt
    assert "Returned products must be unused." in prompt
    assert query in prompt