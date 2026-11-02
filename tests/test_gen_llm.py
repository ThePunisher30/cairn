"""Phase 7, steps 1 and 2: calling a language model robustly (`generation/retry.py`, `clients.py`).

The types (`Completion`, `Usage`, `LLMClient`), the errors (`generation/errors.py`) and the scripted
`FakeLLM` are provided. You write:

retry.py
    RetryPolicy(max_attempts=4, base_delay=0.5, max_delay=30.0, jitter=True)   a frozen dataclass;
        ValueError for max_attempts < 1, base_delay < 0, or max_delay < base_delay.

    classify_error(exc) -> LLMError    decide what kind of failure an exception is. Never raises.
        - an LLMError is returned as it is (the same object)
        - the status is `exc.status_code`, or else `exc.response.status_code` (when it is an int):
              429 -> RateLimitError, with retry_after = float(exc.response.headers["retry-after"])
                     when that header exists and is a number, else None
              408 -> LLMTimeout;  500 and above (529 too) -> ServerError
              400, 422 -> BadRequest;  401, 403 -> AuthError;  404 -> NotFound
              any other status -> FatalError
          the result carries `status_code`, and its message is str(exc)
        - no status: a TimeoutError, or a class whose name contains "Timeout" -> LLMTimeout;
          a ConnectionError, or a name containing "Connect" -> ConnectionFailure
        - anything else -> FatalError (a bug in our code must not be retried)

    backoff_delay(attempt, policy, rng=random.random, retry_after=None) -> float
        attempt 0 is the wait after the first failure.
        cap = min(max_delay, base_delay * 2**attempt).
        With jitter the delay is rng() * cap ("full jitter"), without it the cap. A server's
        `retry_after` can only make the wait longer, and the result is never over max_delay.

    call_with_retries(fn, policy=None, *, sleep=time.sleep, rng=random.random, on_retry=None)
        `policy` None means RetryPolicy(). Calls fn() and returns its result. When it raises (an
        Exception; KeyboardInterrupt and the like pass straight through) the error is classified.
        A non-retryable error, or the max_attempts-th failure, is raised: always an LLMError,
        `raise error from exc` when it was converted (the same object when exc already was one).
        Otherwise sleep(backoff_delay(...)) and call again; before each sleep
        `on_retry(attempt, error, delay)` is called (attempt = 1 after the first failure).

    RetryingClient(inner, policy=None, *, sleep=time.sleep, rng=random.random)
        Wraps any LLMClient (a None policy is RetryPolicy()). `.retries` counts the retries made.
        `.last_usage` is the inner one's, and so is `.model` ("" when the inner client has none).
        complete(): call_with_retries around inner.complete.
        stream(): the same, as long as no piece has been yielded yet. After the first piece the
        caller already has part of the answer, so a failure is raised (classified), not retried.

clients.py
    AnthropicClient(model="claude-opus-5-5", *, api_key=None, timeout=60.0, effort="low", sdk=None)
        `sdk` is an object like `anthropic.Anthropic()`; when None the real one is built with
        `max_retries=0` (the retries are ours, one layer, not two) and the package is imported
        then (ImportError mentioning `.[llm]` if it is missing).
        complete(): sdk.messages.create(model=, max_tokens=, system=, messages=,
        output_config={"effort": effort}) and nothing else: NO temperature, top_p or top_k (the API
        rejects them); no output_config when effort is None. The text is the text blocks joined
        (thinking blocks are skipped). Errors are classified (`raise error from exc`).
        stream(): `with sdk.messages.stream(**the same) as s`, yields `s.text_stream`, and sets
        last_usage from `s.get_final_message()`; errors classified.
    OllamaClient(model="llama3.2", *, base_url="http://localhost:11434", timeout=120.0, client=None)
        `client` is an httpx.Client (built from base_url and timeout when None). POST /api/chat with
        {"model", "messages": [system message, *messages], "stream", "options": {"num_predict":
        max_tokens}}. The reply's message.content is the text, prompt_eval_count and eval_count the
        usage (0 when missing), done_reason "stop" -> "end_turn", "length" -> "max_tokens".
        stream(): one JSON object per line, the last one has "done": true and the counts.
        An HTTP error status is classified like any other.
"""

import sys
from types import SimpleNamespace

import pytest

from cairn.generation.clients import AnthropicClient, OllamaClient
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
from cairn.generation.fake import FakeLLM, Partial
from cairn.generation.retry import (
    RetryingClient,
    RetryPolicy,
    backoff_delay,
    call_with_retries,
    classify_error,
)
from cairn.generation.types import Completion, Usage


class StatusError(Exception):
    """Looks like the status errors of the provider SDKs (`.status_code`, `.response.headers`)."""

    def __init__(self, status_code, headers=None, message="boom"):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers or {}, status_code=status_code)


class APITimeoutError(Exception):
    pass


class ConnectError(Exception):
    pass


NO_JITTER = RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=30.0, jitter=False)


# ------------------------------------------------------------------------------- policy
def test_the_default_policy():
    p = RetryPolicy()
    assert (p.max_attempts, p.base_delay, p.max_delay, p.jitter) == (4, 0.5, 30.0, True)


@pytest.mark.parametrize(
    "arguments",
    [{"max_attempts": 0}, {"base_delay": -1}, {"base_delay": 10, "max_delay": 5}],
)
def test_a_bad_policy_is_refused(arguments):
    with pytest.raises(ValueError):
        RetryPolicy(**arguments)


# --------------------------------------------------------------------- classify_error
@pytest.mark.parametrize(
    "status, expected",
    [
        (429, RateLimitError),
        (408, LLMTimeout),
        (500, ServerError),
        (503, ServerError),
        (529, ServerError),
        (400, BadRequest),
        (422, BadRequest),
        (401, AuthError),
        (403, AuthError),
        (404, NotFound),
        (418, FatalError),
        (409, FatalError),
    ],
)
def test_a_status_decides_the_kind_of_error(status, expected):
    error = classify_error(StatusError(status))
    assert type(error) is expected
    assert error.status_code == status


@pytest.mark.parametrize("status", [429, 408, 500, 529])
def test_those_statuses_are_worth_retrying(status):
    assert classify_error(StatusError(status)).retryable is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 418])
def test_those_statuses_are_not(status):
    assert classify_error(StatusError(status)).retryable is False


def test_the_message_is_kept():
    assert "the model is busy" in str(classify_error(StatusError(529, message="the model is busy")))


def test_a_rate_limit_carries_the_retry_after_header():
    assert classify_error(StatusError(429, {"retry-after": "7"})).retry_after == 7.0
    assert classify_error(StatusError(429, {"retry-after": "2.5"})).retry_after == 2.5


def test_a_missing_or_unreadable_retry_after_is_none():
    assert classify_error(StatusError(429)).retry_after is None
    assert classify_error(StatusError(429, {"retry-after": "soon"})).retry_after is None


def test_a_rate_limit_without_a_response_or_without_headers_still_classifies():
    class NoResponse(Exception):
        status_code = 429

    class NoHeaders(Exception):
        status_code = 429
        response = SimpleNamespace(status_code=429)

    for exc in (NoResponse("x"), NoHeaders("x")):
        error = classify_error(exc)
        assert type(error) is RateLimitError and error.retry_after is None


def test_the_status_can_sit_on_the_response_only_like_httpx_errors():
    exc = Exception("bad gateway")
    exc.response = SimpleNamespace(status_code=502, headers={})
    assert type(classify_error(exc)) is ServerError


def test_timeouts_are_found_by_type_or_by_name():
    assert type(classify_error(TimeoutError("slow"))) is LLMTimeout
    assert type(classify_error(APITimeoutError("slow"))) is LLMTimeout


def test_subclasses_of_the_builtin_errors_are_found_by_type():
    class Slow(TimeoutError):
        pass

    class Reset(ConnectionError):
        pass

    assert type(classify_error(Slow("x"))) is LLMTimeout
    assert type(classify_error(Reset("x"))) is ConnectionFailure


def test_connection_problems_are_found_by_type_or_by_name():
    assert type(classify_error(ConnectionError("reset"))) is ConnectionFailure
    assert type(classify_error(ConnectError("refused"))) is ConnectionFailure
    assert classify_error(ConnectError("refused")).retryable is True


def test_a_bug_in_our_own_code_is_not_retried():
    error = classify_error(ValueError("my mistake"))
    assert type(error) is FatalError and error.retryable is False


def test_an_llm_error_is_returned_as_it_is():
    original = RateLimitError("slow down")
    assert classify_error(original) is original


# ------------------------------------------------------------------------ backoff_delay
def test_without_jitter_the_delay_doubles_up_to_the_cap():
    policy = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=False)
    assert [backoff_delay(n, policy) for n in range(6)] == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]


def test_with_jitter_the_delay_is_a_random_share_of_the_cap():
    policy = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=True)
    assert backoff_delay(2, policy, rng=lambda: 0.5) == 2.0  # half of 4
    assert backoff_delay(2, policy, rng=lambda: 0.0) == 0.0
    assert backoff_delay(2, policy, rng=lambda: 1.0) == 4.0


def test_the_cap_is_applied_before_the_jitter():
    policy = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=True)
    assert backoff_delay(20, policy, rng=lambda: 0.5) == 5.0  # half of 10, not of 2**20


def test_a_server_retry_after_can_only_lengthen_the_wait():
    assert backoff_delay(0, NO_JITTER, retry_after=5.0) == 5.0
    assert backoff_delay(3, NO_JITTER, retry_after=2.0) == 8.0


def test_the_wait_never_goes_over_max_delay():
    assert backoff_delay(0, NO_JITTER, retry_after=500.0) == 30.0


# -------------------------------------------------------------------- call_with_retries
class Flaky:
    """Fails with the given errors in turn, then returns "ok"."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def test_success_on_the_first_try_never_sleeps():
    sleeps = []
    assert call_with_retries(Flaky(), NO_JITTER, sleep=sleeps.append) == "ok"
    assert sleeps == []


def test_a_retryable_failure_is_retried_after_the_backoff():
    fn, sleeps = Flaky(StatusError(500), StatusError(503)), []
    assert call_with_retries(fn, NO_JITTER, sleep=sleeps.append) == "ok"
    assert fn.calls == 3
    assert sleeps == [1.0, 2.0]


def test_on_retry_hears_about_every_retry():
    seen = []
    call_with_retries(
        Flaky(StatusError(500), StatusError(500)),
        NO_JITTER,
        sleep=lambda s: None,
        on_retry=lambda attempt, error, delay: seen.append((attempt, type(error), delay)),
    )
    assert seen == [(1, ServerError, 1.0), (2, ServerError, 2.0)]


def test_a_fatal_error_is_raised_at_once():
    fn, sleeps = Flaky(StatusError(401)), []
    with pytest.raises(AuthError):
        call_with_retries(fn, NO_JITTER, sleep=sleeps.append)
    assert fn.calls == 1 and sleeps == []


def test_when_the_attempts_run_out_the_last_error_is_raised():
    fn, sleeps = Flaky(*[StatusError(429)] * 10), []
    policy = RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=30.0, jitter=False)
    with pytest.raises(RateLimitError):
        call_with_retries(fn, policy, sleep=sleeps.append)
    assert fn.calls == 3
    assert sleeps == [1.0, 2.0]  # no sleep after the last failure


def test_one_attempt_means_no_retry():
    fn = Flaky(StatusError(500))
    with pytest.raises(ServerError):
        call_with_retries(fn, RetryPolicy(max_attempts=1), sleep=lambda s: None)
    assert fn.calls == 1


def test_the_retry_after_of_a_rate_limit_is_waited():
    sleeps = []
    call_with_retries(Flaky(StatusError(429, {"retry-after": "5"})), NO_JITTER, sleep=sleeps.append)
    assert sleeps == [5.0]


def test_an_unknown_exception_is_not_retried_and_is_raised_as_an_llm_error():
    fn = Flaky(ValueError("bug"))
    with pytest.raises(FatalError) as caught:
        call_with_retries(fn, NO_JITTER, sleep=lambda s: None)
    assert fn.calls == 1
    assert isinstance(caught.value.__cause__, ValueError)


def test_a_converted_error_remembers_what_caused_it():
    original = StatusError(404)
    with pytest.raises(NotFound) as caught:
        call_with_retries(Flaky(original), NO_JITTER, sleep=lambda s: None)
    assert caught.value.__cause__ is original


def test_an_llm_error_is_raised_as_the_same_object():
    original = ServerError("down")
    with pytest.raises(ServerError) as caught:
        call_with_retries(Flaky(*[original] * 5), RetryPolicy(max_attempts=2), sleep=lambda s: None)
    assert caught.value is original


def test_keyboard_interrupt_is_not_swallowed():
    fn = Flaky(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        call_with_retries(fn, NO_JITTER, sleep=lambda s: None)
    assert fn.calls == 1


# ---------------------------------------------------------------------- RetryingClient
MESSAGES = [{"role": "user", "content": "hi"}]


def retrying(inner, **options):
    sleeps = []
    client = RetryingClient(inner, NO_JITTER, sleep=sleeps.append, **options)
    return client, sleeps


def test_a_call_that_works_passes_through():
    inner = FakeLLM(["hello"])
    client, sleeps = retrying(inner)
    result = client.complete("sys", MESSAGES, max_tokens=77)
    assert result.text == "hello"
    assert inner.calls[0]["system"] == "sys" and inner.calls[0]["max_tokens"] == 77
    assert (client.retries, sleeps) == (0, [])
    assert client.last_usage == Usage(10, 5)


def test_the_model_name_and_the_usage_are_the_inner_clients():
    client, _ = retrying(FakeLLM(["x"], model="m-1"))
    assert client.model == "m-1"
    assert RetryingClient(SimpleNamespace(last_usage=None)).model == ""


def test_complete_retries_a_transient_failure():
    client, sleeps = retrying(FakeLLM([ServerError("busy"), RateLimitError("slow"), "finally"]))
    assert client.complete("s", MESSAGES).text == "finally"
    assert client.retries == 2 and sleeps == [1.0, 2.0]


def test_complete_does_not_retry_a_fatal_error():
    inner = FakeLLM([AuthError("bad key"), "never"])
    client, _ = retrying(inner)
    with pytest.raises(AuthError):
        client.complete("s", MESSAGES)
    assert len(inner.calls) == 1 and client.retries == 0


def test_stream_retries_a_failure_before_the_first_piece():
    inner = FakeLLM([ServerError("busy"), "a b c"])
    client, sleeps = retrying(inner)
    assert "".join(client.stream("s", MESSAGES)) == "a b c"
    assert client.retries == 1 and len(inner.calls) == 2 and sleeps == [1.0]
    assert client.last_usage == Usage(10, 5)


def test_stream_does_not_retry_after_the_first_piece():
    inner = FakeLLM([Partial("one two ", ServerError("dropped")), "never used"])
    client, _ = retrying(inner)
    pieces = []
    with pytest.raises(ServerError):
        for piece in client.stream("s", MESSAGES):
            pieces.append(piece)
    assert "".join(pieces) == "one two "
    assert len(inner.calls) == 1 and client.retries == 0


def test_stream_raises_a_converted_error_after_the_first_piece():
    client, _ = retrying(FakeLLM([Partial("one ", ValueError("bug"))]))
    with pytest.raises(FatalError):
        list(client.stream("s", MESSAGES))


def test_stream_gives_up_after_max_attempts():
    inner = FakeLLM([ServerError("a"), ServerError("b"), ServerError("c")])
    client = RetryingClient(
        inner, RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0), sleep=lambda s: None
    )
    with pytest.raises(ServerError):
        list(client.stream("s", MESSAGES))
    assert len(inner.calls) == 3


def test_stream_does_not_retry_a_fatal_error():
    inner = FakeLLM([NotFound("no such model"), "never"])
    client, _ = retrying(inner)
    with pytest.raises(NotFound):
        list(client.stream("s", MESSAGES))
    assert len(inner.calls) == 1


# --------------------------------------------------------------------- AnthropicClient
class Block:
    def __init__(self, type, text=""):
        self.type, self.text = type, text


def response(text="hello", stop_reason="end_turn", inp=11, out=7, model="claude-test"):
    return SimpleNamespace(
        content=[Block("thinking", "hmm"), Block("text", text), Block("text", " there")],
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out),
        model=model,
        stop_reason=stop_reason,
    )


class FakeStream:
    def __init__(self, pieces, final, error=None):
        self.text_stream = self._pieces(pieces, error)
        self._final = final

    @staticmethod
    def _pieces(pieces, error):
        yield from pieces
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_final_message(self):
        return self._final


class FakeSDK:
    def __init__(self, reply=None, error=None, stream=None):
        self.calls, self.stream_calls = [], []
        self._reply, self._error, self._stream = reply, error, stream
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._reply

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._stream


def test_complete_returns_the_text_blocks_usage_and_model():
    client = AnthropicClient(sdk=FakeSDK(response()))
    result = client.complete("be brief", MESSAGES, max_tokens=300)
    assert isinstance(result, Completion)
    assert result.text == "hello there"  # the thinking block is skipped
    assert result.usage == Usage(11, 7) and client.last_usage == Usage(11, 7)
    assert result.model == "claude-test" and result.stop_reason == "end_turn"
    assert result.latency_ms >= 0


def test_the_request_has_exactly_the_documented_fields():
    sdk = FakeSDK(response())
    AnthropicClient(sdk=sdk).complete("be brief", MESSAGES, max_tokens=300)
    assert sdk.calls == [
        {
            "model": "claude-opus-5-5",
            "max_tokens": 300,
            "system": "be brief",
            "messages": MESSAGES,
            "output_config": {"effort": "low"},
        }
    ]


def test_sampling_parameters_are_never_sent():
    sdk = FakeSDK(response())
    AnthropicClient(model="m", effort="high", sdk=sdk).complete("s", MESSAGES)
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in sdk.calls[0]
    assert sdk.calls[0]["model"] == "m"
    assert sdk.calls[0]["output_config"] == {"effort": "high"}


def test_no_effort_means_no_output_config():
    sdk = FakeSDK(response())
    AnthropicClient(effort=None, sdk=sdk).complete("s", MESSAGES)
    assert "output_config" not in sdk.calls[0]


@pytest.mark.parametrize("reason", ["max_tokens", "refusal", "stop_sequence"])
def test_the_stop_reason_is_passed_through(reason):
    result = AnthropicClient(sdk=FakeSDK(response(stop_reason=reason))).complete("s", MESSAGES)
    assert result.stop_reason == reason


def test_a_missing_stop_reason_means_the_turn_ended():
    result = AnthropicClient(sdk=FakeSDK(response(stop_reason=None))).complete("s", MESSAGES)
    assert result.stop_reason == "end_turn"


def test_a_failing_call_is_classified_and_not_retried_here():
    original = StatusError(429, {"retry-after": "3"})
    sdk = FakeSDK(error=original)
    with pytest.raises(RateLimitError) as caught:
        AnthropicClient(sdk=sdk).complete("s", MESSAGES)
    assert caught.value.retry_after == 3.0
    assert caught.value.__cause__ is original
    assert len(sdk.calls) == 1


def test_a_timeout_from_the_sdk_is_a_timeout():
    with pytest.raises(LLMTimeout):
        AnthropicClient(sdk=FakeSDK(error=APITimeoutError("slow"))).complete("s", MESSAGES)


def test_stream_yields_the_pieces_and_sets_the_usage_at_the_end():
    sdk = FakeSDK(stream=FakeStream(["Hel", "lo ", "you"], response(inp=20, out=3)))
    client = AnthropicClient(sdk=sdk)
    stream = client.stream("be brief", MESSAGES, max_tokens=50)
    assert client.last_usage is None  # nothing has been read yet
    assert list(stream) == ["Hel", "lo ", "you"]
    assert client.last_usage == Usage(20, 3)
    assert sdk.stream_calls[0]["max_tokens"] == 50
    assert sdk.stream_calls[0]["output_config"] == {"effort": "low"}
    assert "temperature" not in sdk.stream_calls[0]


def test_stream_classifies_an_error_when_the_stream_cannot_be_opened():
    with pytest.raises(ServerError):
        list(AnthropicClient(sdk=FakeSDK(error=StatusError(500))).stream("s", MESSAGES))


def test_stream_classifies_an_error_in_the_middle():
    sdk = FakeSDK(stream=FakeStream(["a "], response(), error=StatusError(529)))
    pieces = []
    with pytest.raises(ServerError):
        for piece in AnthropicClient(sdk=sdk).stream("s", MESSAGES):
            pieces.append(piece)
    assert pieces == ["a "]


def test_the_real_sdk_is_built_with_its_own_retries_switched_off(monkeypatch):
    built = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    AnthropicClient("m", api_key="k", timeout=12.0)
    assert built == {"api_key": "k", "timeout": 12.0, "max_retries": 0}


def test_without_the_package_the_error_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` fail
    with pytest.raises(ImportError, match=r"\.\[llm\]"):
        AnthropicClient()


# ----------------------------------------------------------------------- OllamaClient
httpx = pytest.importorskip("httpx")


def ollama(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama.test")
    return OllamaClient("llama-test", client=client)


def json_reply(**data):
    return httpx.Response(200, json=data)


def test_ollama_complete_sends_the_chat_request_and_reads_the_reply():
    seen = []

    def handler(request):
        seen.append(request)
        return json_reply(
            model="llama-test",
            message={"role": "assistant", "content": "hi there"},
            done=True,
            done_reason="stop",
            prompt_eval_count=12,
            eval_count=4,
        )

    client = ollama(handler)
    result = client.complete("be brief", MESSAGES, max_tokens=64)
    assert result.text == "hi there"
    assert result.usage == Usage(12, 4) and client.last_usage == Usage(12, 4)
    assert result.stop_reason == "end_turn" and result.model == "llama-test"
    request = seen[0]
    assert (request.method, request.url.path) == ("POST", "/api/chat")
    import json

    body = json.loads(request.content)
    assert body["model"] == "llama-test" and body["stream"] is False
    assert body["options"] == {"num_predict": 64}
    assert body["messages"] == [{"role": "system", "content": "be brief"}, *MESSAGES]


def test_ollama_length_means_max_tokens_and_missing_counts_mean_zero():
    client = ollama(
        lambda r: json_reply(message={"content": "cut"}, done=True, done_reason="length")
    )
    result = client.complete("s", MESSAGES)
    assert result.stop_reason == "max_tokens" and result.usage == Usage(0, 0)


@pytest.mark.parametrize(
    "status, expected",
    [(500, ServerError), (404, NotFound), (400, BadRequest), (401, AuthError)],
)
def test_ollama_http_errors_are_classified(status, expected):
    client = ollama(lambda r: httpx.Response(status, json={"error": "nope"}))
    with pytest.raises(expected):
        client.complete("s", MESSAGES)


def test_ollama_rate_limit_keeps_the_retry_after():
    client = ollama(lambda r: httpx.Response(429, headers={"retry-after": "4"}))
    with pytest.raises(RateLimitError) as caught:
        client.complete("s", MESSAGES)
    assert caught.value.retry_after == 4.0


def test_ollama_network_failures_are_retryable():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ConnectionFailure):
        ollama(refuse).complete("s", MESSAGES)
    with pytest.raises(LLMTimeout):
        ollama(slow).complete("s", MESSAGES)


def test_ollama_stream_yields_the_pieces_and_reads_the_counts_from_the_last_line():
    lines = [
        {"message": {"content": "Hel"}, "done": False},
        {"message": {"content": "lo"}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 9, "eval_count": 2},
    ]
    import json

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        body = "\n".join(json.dumps(line) for line in lines) + "\n\n"
        return httpx.Response(200, content=body.encode())

    client = ollama(handler)
    assert list(client.stream("s", MESSAGES, max_tokens=30)) == ["Hel", "lo"]
    assert client.last_usage == Usage(9, 2)
    assert seen[0]["stream"] is True and seen[0]["options"] == {"num_predict": 30}


def test_ollama_stream_classifies_an_http_error():
    client = ollama(lambda r: httpx.Response(503))
    with pytest.raises(ServerError):
        list(client.stream("s", MESSAGES))


def test_every_error_the_clients_raise_is_an_llm_error():
    for exc in (StatusError(500), APITimeoutError("x"), ValueError("y")):
        with pytest.raises(LLMError):
            AnthropicClient(sdk=FakeSDK(error=exc)).complete("s", MESSAGES)
