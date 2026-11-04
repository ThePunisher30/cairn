import random
import time
from dataclasses import dataclass

from cairn.generation.errors import (
    AuthError,
    BadRequest,
    ConnectionFailure,
    FatalError,
    LLMError,
    LLMTimeout,
    NotFound,
    RateLimitError,
    ServerError,
)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: bool = True

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, it is currently {self.max_attempts}")
        if self.base_delay < 0:
            raise ValueError(f"base_delay must be >= 0, it is currently {self.base_delay}")
        if self.max_delay < self.base_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must not be below base_delay ({self.base_delay})"
            )


def _status_of(exc):
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _retry_after(exc):
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def classify_error(exc):
    if isinstance(exc, LLMError):
        return exc
    message = str(exc)
    status = _status_of(exc)
    if status is not None:
        if status == 429:
            return RateLimitError(message, status_code=status, retry_after=_retry_after(exc))
        if status == 408:
            return LLMTimeout(message, status_code=status)
        if status >= 500:
            return ServerError(message, status_code=status)
        if status in (400, 422):
            return BadRequest(message, status_code=status)
        if status in (401, 403):
            return AuthError(message, status_code=status)
        if status == 404:
            return NotFound(message, status_code=status)
        return FatalError(message, status_code=status)
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or "Timeout" in name:
        return LLMTimeout(message)
    if isinstance(exc, ConnectionError) or "Connect" in name:
        return ConnectionFailure(message)
    return FatalError(message)


# placeholders so that the test file can be imported (the real ones come in the next steps)
def backoff_delay(attempt, policy, rng=random.random, retry_after=None):
    cap = min(policy.max_delay, policy.base_delay * 2**attempt)
    delay = rng() * cap if policy.jitter else cap
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(delay, policy.max_delay)


def call_with_retries(fn, policy=None, *, sleep=time.sleep, rng=random.random, on_retry=None):
    if policy is None:
        policy = RetryPolicy()
    failures = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            error = classify_error(exc)
            failures += 1
            if not error.retryable or failures >= policy.max_attempts:
                if error is exc:
                    raise
                raise error from exc
            delay = backoff_delay(failures - 1, policy, rng=rng, retry_after=error.retry_after)
            if on_retry is not None:
                on_retry(failures, error, delay)
            sleep(delay)


class RetryingClient:
    def __init__(self, inner, policy=None, *, sleep=time.sleep, rng=random.random):
        self._inner = inner
        self._policy = RetryPolicy() if policy is None else policy
        self._sleep = sleep
        self._rng = rng
        self.retries = 0

    @property
    def model(self):
        return getattr(self._inner, "model", "")

    @property
    def last_usage(self):
        return self._inner.last_usage

    def _count_retry(self, attempt, error, delay):
        self.retries += 1

    def complete(self, system, messages, *, max_tokens=1024):
        return call_with_retries(
            lambda: self._inner.complete(system, messages, max_tokens=max_tokens),
            self._policy,
            sleep=self._sleep,
            rng=self._rng,
            on_retry=self._count_retry,
        )

    def stream(self, system, messages, *, max_tokens=1024):
        failures = 0
        while True:
            started = False
            try:
                for piece in self._inner.stream(system, messages, max_tokens=max_tokens):
                    started = True
                    yield piece
                return
            except Exception as exc:
                error = classify_error(exc)
                failures += 1
                if started or not error.retryable or failures >= self._policy.max_attempts:
                    if error is exc:
                        raise
                    raise error from exc
                delay = backoff_delay(
                    failures - 1, self._policy, rng=self._rng, retry_after=error.retry_after
                )
                self.retries += 1
                self._sleep(delay)
