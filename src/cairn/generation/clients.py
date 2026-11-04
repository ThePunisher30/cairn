import json
import time

from cairn.generation.retry import classify_error
from cairn.generation.types import Completion, Usage

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}


def _reraise(exc):
    """Raise `exc` as one of our errors (the same object if it already is one)."""
    error = classify_error(exc)
    if error is exc:
        raise exc
    raise error from exc


def _usage(data):
    return Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0))


def _sdk_usage(message):
    return Usage(message.usage.input_tokens, message.usage.output_tokens)


class AnthropicClient:
    def __init__(
        self, model="claude-opus-5-5", *, api_key=None, timeout=60.0, effort="low", sdk=None
    ):
        if sdk is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    'the anthropic package is missing: pip install -e ".[llm]"'
                ) from exc
            sdk = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        self.effort = effort
        self.sdk = sdk
        self.last_usage = None

    def _request(self, system, messages, max_tokens):
        request = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if self.effort is not None:
            request["output_config"] = {"effort": self.effort}
        return request

    def complete(self, system, messages, *, max_tokens=1024):
        started = time.perf_counter()
        try:
            message = self.sdk.messages.create(**self._request(system, messages, max_tokens))
        except Exception as exc:
            _reraise(exc)
        latency_ms = (time.perf_counter() - started) * 1000
        self.last_usage = _sdk_usage(message)
        return Completion(
            text="".join(block.text for block in message.content if block.type == "text"),
            usage=self.last_usage,
            model=message.model,
            stop_reason=message.stop_reason or "end_turn",
            latency_ms=latency_ms,
        )

    def stream(self, system, messages, *, max_tokens=1024):
        try:
            with self.sdk.messages.stream(**self._request(system, messages, max_tokens)) as s:
                yield from s.text_stream
                self.last_usage = _sdk_usage(s.get_final_message())
        except Exception as exc:
            _reraise(exc)


class OllamaClient:
    def __init__(
        self,
        model="llama3.2",
        *,
        base_url="http://localhost:11434",
        timeout=120.0,
        think=None,
        client=None,
    ):
        if client is None:
            import httpx

            client = httpx.Client(base_url=base_url, timeout=timeout)
        self.model = model
        self.think = think
        self.client = client
        self.last_usage = None

    def _body(self, system, messages, max_tokens, stream):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": stream,
            "options": {"num_predict": max_tokens},
        }
        if self.think is not None:
            body["think"] = self.think
        return body

    def complete(self, system, messages, *, max_tokens=1024):
        body = self._body(system, messages, max_tokens, False)
        started = time.perf_counter()
        try:
            response = self.client.post("/api/chat", json=body)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            _reraise(exc)
        latency_ms = (time.perf_counter() - started) * 1000
        self.last_usage = _usage(data)
        reason = data.get("done_reason", "stop")
        return Completion(
            text=(data.get("message") or {}).get("content") or "",
            usage=self.last_usage,
            model=data.get("model") or self.model,
            stop_reason=STOP_REASONS.get(reason, reason),
            latency_ms=latency_ms,
        )

    def stream(self, system, messages, *, max_tokens=1024):
        body = self._body(system, messages, max_tokens, True)
        try:
            with self.client.stream("POST", "/api/chat", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    piece = (data.get("message") or {}).get("content") or ""
                    if piece:
                        yield piece
                    if data.get("done"):
                        self.last_usage = _usage(data)
        except Exception as exc:
            _reraise(exc)
