# ADR 0001: Reference documents by server-managed IDs

- **Status:** Accepted (Stage 0). Superseded by the database-backed registry planned for Stage 1.
- **Date:** 2026-10-06

## Context
`QuestionRequest` had a `file_path: str` field. Once an endpoint accepted it, any client could make the
server read any file it can access (`../../etc/passwd`, `.env`, SSH keys). This is a path traversal /
arbitrary file read vulnerability (CWE-22). For a product sold to privacy-sensitive businesses
(§203 StGB professions, DSGVO), this is disqualifying.

## Decision
- Clients send only a `document_id`. The server owns the mapping from ID to storage.
- In Stage 0 the mapping is `<COPILOT_DOCUMENTS_DIR>/<id>.txt` (`DocumentRegistry`):
  - IDs must match `^[a-z0-9][a-z0-9_-]{0,63}$`. This rejects separators, dots, `..`, absolute
    paths, NUL bytes and Unicode look-alikes before any filesystem access.
  - The candidate path is resolved and must stay inside the resolved base directory, which also
    rejects symlinks that point outside it.
- `QuestionRequest` uses `extra="forbid"`, so a client that still sends `file_path` gets a validation
  error instead of having the field silently ignored.
- Chunking parameters (`chunk_size`, `overlap`) were removed from the request. They are server
  settings, because from Stage 1 on chunking happens once at ingest time, not per query.
- Error messages include the document ID or the file name, never the server's directory layout.

## Alternatives considered
| Option | Why not |
|---|---|
| Allow-list of absolute paths | It still lets the client name paths, and is easy to get wrong when the list grows. |
| Sanitise the path with `os.path.basename` | It does not stop symlinks and is easy to bypass on some platforms. A positive ID pattern is stricter. |
| Database table now | That is Stage 1 scope, together with ingestion and pgvector. Not needed to close the hole. |

## Trade-offs
- Only flat `.txt` files are supported until Stage 1 adds ingestion (PDF, DOCX, HTML) and persistence.
- IDs are file names, so renaming a file changes its ID. This is acceptable for a pre-pilot.
- There is no tenant scoping yet. Multi-tenancy and query-level isolation arrive in Stage 2.
