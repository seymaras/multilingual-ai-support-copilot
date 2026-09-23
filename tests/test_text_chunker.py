from multilingual_support_copilot.text_chunker import chunk_text

def test_chunk_text_splits_text_by_chunk_size() -> None:
    text = "one two three four five"

    result = chunk_text(text, chunk_size=2)

    assert result == ["one two", "three four", "five"]


def test_chunk_text_returns_empty_list_for_empty_text() -> None:
    result = chunk_text("")

    assert result == []


def test_chunk_text_returns_single_chunk_when_text_is_short() -> None:
    text = "hello world"

    result = chunk_text(text, chunk_size=50)

    assert result == ["hello world"]

def test_chunk_text_with_overlap() -> None:
    text = "one two three four five six seven eight"

    result = chunk_text(
        text,
        chunk_size=4,
        overlap=2,
    )

    assert result == [
        "one two three four",
        "three four five six",
        "five six seven eight",
    ]