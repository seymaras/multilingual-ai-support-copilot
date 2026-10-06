import os
from pathlib import Path

import pytest

from multilingual_support_copilot.document_loader import load_text_file
from multilingual_support_copilot.errors import (
    DocumentLoadError,
    DocumentNotFoundError,
    EmptyDocumentError,
)

REPO_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_load_text_file_reads_content() -> None:
    result = load_text_file(REPO_DATA_DIR / "sample.txt")

    expected = (
        "Our customer support team is available Monday through Friday.\n"
        "Customers can contact support by email."
    )

    assert result == expected


def test_load_text_file_raises_not_found_for_missing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "missing_file.txt"

    with pytest.raises(DocumentNotFoundError) as exc_info:
        load_text_file(file_path)

    assert isinstance(exc_info.value, FileNotFoundError)
    assert "missing_file.txt" in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("content", ["", "   \n\t  \n"])
def test_load_text_file_rejects_empty_content(tmp_path: Path, content: str) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text(content, encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        load_text_file(file_path)


def test_load_text_file_rejects_invalid_utf8(tmp_path: Path) -> None:
    file_path = tmp_path / "latin1.txt"
    file_path.write_bytes(b"caf\xe9 \xff\xfe")

    with pytest.raises(DocumentLoadError, match="not valid UTF-8") as exc_info:
        load_text_file(file_path)

    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)
    assert str(tmp_path) not in str(exc_info.value)


def test_load_text_file_rejects_directory(tmp_path: Path) -> None:
    directory = tmp_path / "folder.txt"
    directory.mkdir()

    with pytest.raises(DocumentLoadError):
        load_text_file(directory)


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="permission checks do not apply to root",
)
def test_load_text_file_rejects_unreadable_file(tmp_path: Path) -> None:
    file_path = tmp_path / "locked.txt"
    file_path.write_text("secret", encoding="utf-8")
    file_path.chmod(0o000)
    try:
        with pytest.raises(DocumentLoadError) as exc_info:
            load_text_file(file_path)
    finally:
        file_path.chmod(0o600)

    assert isinstance(exc_info.value.__cause__, PermissionError)
    assert str(tmp_path) not in str(exc_info.value)
