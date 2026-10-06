"""Load plain-text documents from disk with domain-specific errors."""

from pathlib import Path

from multilingual_support_copilot.errors import (
    DocumentLoadError,
    DocumentNotFoundError,
    EmptyDocumentError,
)


def load_text_file(file_path: Path) -> str:
    """Read ``file_path`` as UTF-8 text.

    Error messages contain only the file name, never the full server path.

    Raises:
        DocumentNotFoundError: the file does not exist.
        DocumentLoadError: the path is a directory, is not readable, or is not valid UTF-8.
        EmptyDocumentError: the file contains no non-whitespace text.
    """
    name = file_path.name
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as err:
        raise DocumentNotFoundError(f"Document file not found: {name}") from err
    except IsADirectoryError as err:
        raise DocumentLoadError(f"Document path is a directory: {name}") from err
    except PermissionError as err:
        raise DocumentLoadError(f"Permission denied reading document: {name}") from err
    except UnicodeDecodeError as err:
        raise DocumentLoadError(f"Document file is not valid UTF-8: {name}") from err

    if not text.strip():
        raise EmptyDocumentError(f"Document file is empty: {name}")
    return text
