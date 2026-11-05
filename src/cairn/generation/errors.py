"""The errors a language-model call can end in (Phase 7). Provided, not part of the exercise.

The point of the split is the `retryable` flag: a rate limit or a server hiccup may work if you try
again a moment later; a bad request or a wrong API key never will, and retrying it only wastes time.
"""


class LLMError(Exception):
    retryable = False

    def __init__(self, message: str = "", *, status_code: int | None = None, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after  # seconds the server asked us to wait, or None


class RetryableError(LLMError):
    retryable = True


class RateLimitError(RetryableError):
    """HTTP 429."""


class ServerError(RetryableError):
    """HTTP 5xx, including 529 (overloaded)."""


class LLMTimeout(RetryableError):
    """The call took too long (a client timeout, or HTTP 408)."""


class ConnectionFailure(RetryableError):
    """The network failed before an answer came."""


class FatalError(LLMError):
    """Retrying cannot help."""


class BadRequest(FatalError):
    """HTTP 400 or 422: the request itself is wrong."""


class AuthError(FatalError):
    """HTTP 401 or 403: a wrong or missing API key, or no permission."""


class NotFound(FatalError):
    """HTTP 404: a wrong model name or address."""
