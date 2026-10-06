"""Server-managed document registry.

Clients reference documents only by ID. An ID maps to ``<documents_dir>/<id>.txt``;
the resolved path must stay inside ``documents_dir`` (this also rejects symlinks
pointing elsewhere). Stage 1 replaces this with a database-backed registry.
"""

import re
from pathlib import Path

from multilingual_support_copilot.config import get_settings
from multilingual_support_copilot.errors import DocumentNotFoundError, InvalidDocumentIdError

DOCUMENT_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
_DOCUMENT_ID_RE = re.compile(DOCUMENT_ID_PATTERN)
_SUFFIX = ".txt"


class DocumentRegistry:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir.resolve()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def list_ids(self) -> list[str]:
        if not self._base_dir.is_dir():
            return []
        return sorted(
            path.stem
            for path in self._base_dir.glob(f"*{_SUFFIX}")
            if _DOCUMENT_ID_RE.fullmatch(path.stem)
            and self._is_inside_base(path)
            and path.resolve().is_file()
        )

    def resolve(self, document_id: str) -> Path:
        if not _DOCUMENT_ID_RE.fullmatch(document_id):
            raise InvalidDocumentIdError(f"Invalid document ID: {document_id!r}")
        candidate = self._base_dir / f"{document_id}{_SUFFIX}"
        if not self._is_inside_base(candidate) or not candidate.resolve().is_file():
            raise DocumentNotFoundError(f"Unknown document ID: {document_id!r}")
        return candidate.resolve()

    def _is_inside_base(self, path: Path) -> bool:
        return path.resolve().is_relative_to(self._base_dir)


def get_document_registry() -> DocumentRegistry:
    return DocumentRegistry(get_settings().documents_dir)
