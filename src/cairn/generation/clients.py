import json
import time

from cairn.generation.retry import classify_error
from cairn.generation.types import Completion, ToolCall, Usage

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


def _anthropic_messages(messages):
    """Our neutral messages as the Messages API wants them. An assistant turn that called tools
    becomes text + tool_use blocks; the tool results of one turn are merged into ONE user
    message of tool_result blocks. The caller's list and dicts are not changed."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            blocks = [{"type": "text", "text": m["content"]}] if m["content"] else []
            blocks += [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                for c in m["tool_calls"]
            ]
            out.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m["content"],
            }
            last = out[-1] if out else None
            if last and last["role"] == "user" and isinstance(last["content"], list):
                # user messages in our format have text content, so a list here is one we built
                last["content"].append(result)
            else:
                out.append({"role": "user", "content": [result]})
        else:
            out.append(m)
    return out


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

    def _request(self, system, messages, max_tokens, tools=None):
        request = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": _anthropic_messages(messages),
        }
        if tools is not None:
            request["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        if self.effort is not None:
            request["output_config"] = {"effort": self.effort}
        return request

    def complete(self, system, messages, *, max_tokens=1024, tools=None):
        started = time.perf_counter()
        try:
            message = self.sdk.messages.create(**self._request(system, messages, max_tokens, tools))
        except Exception as exc:
            _reraise(exc)
        latency_ms = (time.perf_counter() - started) * 1000
        self.last_usage = _sdk_usage(message)
        calls = tuple(
            ToolCall(block.id, block.name, dict(block.input))
            for block in message.content
            if block.type == "tool_use"
        )
        return Completion(
            text="".join(block.text for block in message.content if block.type == "text"),
            usage=self.last_usage,
            model=message.model,
            stop_reason=message.stop_reason or "end_turn",
            latency_ms=latency_ms,
            tool_calls=calls,
        )

    def stream(self, system, messages, *, max_tokens=1024):
        try:
            with self.sdk.messages.stream(**self._request(system, messages, max_tokens)) as s:
                yield from s.text_stream
                self.last_usage = _sdk_usage(s.get_final_message())
        except Exception as exc:
            _reraise(exc)


def _ollama_messages(messages):
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = [
                {"function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in m["tool_calls"]
            ]
            out.append({"role": "assistant", "content": m["content"], "tool_calls": calls})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
        else:
            out.append(m)
    return out


def _ollama_tool_calls(message):
    """The tool calls in an Ollama reply. Arguments may be a dict or a JSON string; anything
    that is not a JSON object becomes {} plus an `error` the agent loop reports to the model."""
    calls = []
    for position, raw in enumerate(message.get("tool_calls") or []):
        fn = raw.get("function") or {}
        args, error = fn.get("arguments", {}), None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        if not isinstance(args, dict):
            args = {}
            error = f"arguments are not a JSON object: {fn.get('arguments')!r}"
        call_id = raw.get("id") or f"call_{position}"
        calls.append(ToolCall(call_id, fn.get("name", ""), args, error))
    return tuple(calls)


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

    def _body(self, system, messages, max_tokens, stream, tools=None):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *_ollama_messages(messages)],
            "stream": stream,
            "options": {"num_predict": max_tokens},
        }
        if tools is not None:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        if self.think is not None:
            body["think"] = self.think
        return body

    def complete(self, system, messages, *, max_tokens=1024, tools=None):
        body = self._body(system, messages, max_tokens, False, tools)
        started = time.perf_counter()
        try:
            response = self.client.post("/api/chat", json=body)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            _reraise(exc)
        latency_ms = (time.perf_counter() - started) * 1000
        self.last_usage = _usage(data)
        message = data.get("message") or {}
        calls = _ollama_tool_calls(message)
        reason = data.get("done_reason", "stop")
        return Completion(
            text=message.get("content") or "",
            usage=self.last_usage,
            model=data.get("model") or self.model,
            stop_reason="tool_use" if calls else STOP_REASONS.get(reason, reason),
            latency_ms=latency_ms,
            tool_calls=calls,
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
