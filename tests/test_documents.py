from pathlib import Path

import pytest

from multilingual_support_copilot.documents import DocumentRegistry, get_document_registry
from multilingual_support_copilot.errors import DocumentNotFoundError, InvalidDocumentIdError

REPO_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _write(path: Path, text: str = "content") -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_resolve_returns_path_for_valid_id(tmp_path: Path) -> None:
    expected = _write(tmp_path / "faq.txt")

    assert DocumentRegistry(tmp_path).resolve("faq") == expected.resolve()


def test_resolve_accepts_digits_hyphens_and_underscores(tmp_path: Path) -> None:
    expected = _write(tmp_path / "0-return_policy.txt")

    assert DocumentRegistry(tmp_path).resolve("0-return_policy") == expected.resolve()


def test_list_ids_is_sorted_and_filters_unusable_entries(tmp_path: Path) -> None:
    _write(tmp_path / "zeta.txt")
    _write(tmp_path / "alpha.txt")
    _write(tmp_path / "notes.md")
    _write(tmp_path / "Upper.txt")
    _write(tmp_path / ".hidden.txt")
    (tmp_path / "subdir").mkdir()
    _write(tmp_path / "subdir" / "nested.txt")
    (tmp_path / "folder.txt").mkdir()

    assert DocumentRegistry(tmp_path).list_ids() == ["alpha", "zeta"]


@pytest.mark.parametrize(
    "document_id",
    [
        "../etc/passwd",
        "..",
        "a/b",
        "a\\b",
        ".hidden",
        "/etc/passwd",
        "Upper",
        "",
        "a" * 65,
        "id.txt",
        "id\x00",
        "id\n",
        "-leading-hyphen",
        "\u0430bc",  # Cyrillic small a
        "\uff46\uff41\uff51",  # fullwidth letters
        "faq\u200b",  # zero-width space
    ],
)
def test_resolve_rejects_invalid_ids(tmp_path: Path, document_id: str) -> None:
    _write(tmp_path / "faq.txt")

    with pytest.raises(InvalidDocumentIdError):
        DocumentRegistry(tmp_path).resolve(document_id)


def test_resolve_accepts_max_length_id(tmp_path: Path) -> None:
    document_id = "a" * 64
    _write(tmp_path / f"{document_id}.txt")

    assert DocumentRegistry(tmp_path).resolve(document_id).name == f"{document_id}.txt"


def test_resolve_missing_id_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(DocumentNotFoundError):
        DocumentRegistry(tmp_path).resolve("missing")


def test_symlink_escaping_base_dir_is_rejected(tmp_path: Path) -> None:
    base = tmp_path / "docs"
    base.mkdir()
    outside = _write(tmp_path / "secret.txt", "secret")
    (base / "escape.txt").symlink_to(outside)
    registry = DocumentRegistry(base)

    with pytest.raises(DocumentNotFoundError):
        registry.resolve("escape")
    assert registry.list_ids() == []


def test_symlinked_base_dir_still_works(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    _write(real / "faq.txt")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    registry = DocumentRegistry(link)

    assert registry.resolve("faq") == (real / "faq.txt").resolve()
    assert registry.list_ids() == ["faq"]


def test_directory_named_like_document_is_not_found(tmp_path: Path) -> None:
    (tmp_path / "x.txt").mkdir()

    with pytest.raises(DocumentNotFoundError):
        DocumentRegistry(tmp_path).resolve("x")


def test_nonexistent_base_dir_lists_nothing(tmp_path: Path) -> None:
    registry = DocumentRegistry(tmp_path / "does-not-exist")

    assert registry.list_ids() == []
    with pytest.raises(DocumentNotFoundError):
        registry.resolve("faq")


def test_get_document_registry_uses_configured_dir(documents_dir: Path) -> None:
    _write(documents_dir / "faq.txt")

    registry = get_document_registry()

    assert registry.base_dir == documents_dir.resolve()
    assert registry.list_ids() == ["faq"]
    assert registry.resolve("faq") == (documents_dir / "faq.txt").resolve()


def test_repo_data_dir_resolves_evaluation_document() -> None:
    registry = DocumentRegistry(REPO_DATA_DIR)

    assert registry.resolve("evaluation_document") == REPO_DATA_DIR / "evaluation_document.txt"
    assert "evaluation_document" in registry.list_ids()
