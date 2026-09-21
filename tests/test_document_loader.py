import pytest
from pathlib import Path

from src.multilingual_support_copilot.document_loader import load_text_file

def test_load_text_file_reads_content() ->None:
    file_path = Path("data/sample.txt")

    result = load_text_file(file_path)

    expected = (
        "Our customer support team is available Monday through Friday.\n"
        "Customers can contact support by email."
    )

    assert result == expected

def test_load_text_file_raises_error_for_missing_file()->None:
    file_path = Path("data/olmayan_dosya.txt")

    with pytest.raises(FileNotFoundError):
        load_text_file(file_path)