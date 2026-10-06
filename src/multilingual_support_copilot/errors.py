"""Domain exceptions. Callers (e.g. the future HTTP layer) map these to responses."""


class CopilotError(Exception):
    """Base class for all copilot errors."""


class InvalidDocumentIdError(CopilotError, ValueError):
    """The document ID does not match the allowed pattern."""


class DocumentNotFoundError(CopilotError, FileNotFoundError):
    """No document exists for the given ID or path."""


class DocumentLoadError(CopilotError):
    """The document exists but could not be read (permissions, encoding)."""


class EmptyDocumentError(CopilotError):
    """The document contains no text."""


class LLMUnavailableError(CopilotError):
    """The LLM backend could not be reached, timed out, or the model is missing."""


class LLMResponseError(CopilotError):
    """The LLM kept returning output that does not match the expected schema."""

    def __init__(self, message: str, *, attempts: int, last_error: Exception | None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
