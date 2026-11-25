import json
import threading

import httpx

from cairn.generation.clients import OllamaClient
from cairn.generation.fake import FakeLLM
from cairn.generation.retry import RetryingClient
from cairn.generation.types import Usage

MESSAGES = [{"role": "user", "content": "q"}]


def ollama_with_usage(tokens_by_prompt):
    """An Ollama client whose server answers each prompt with a different token count."""

    def handler(request):
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        n = tokens_by_prompt[prompt]
        lines = [
            {"message": {"content": "word "}, "done": False},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": n, "eval_count": 1},
        ]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    client = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    return OllamaClient("m", client=client)


def test_a_fork_has_its_own_last_usage_and_shares_the_http_client():
    base = ollama_with_usage({"q": 7})
    fork = base.fork()
    list(fork.stream("s", MESSAGES))
    assert fork.last_usage == Usage(7, 1)
    assert base.last_usage is None  # the shared client's own field was not touched
    assert fork.client is base.client and fork.model == base.model


def test_two_requests_streaming_at_once_each_read_their_own_usage():
    base = ollama_with_usage({"a": 100, "b": 200})
    results = {}

    def run(prompt):
        fork = base.fork()
        list(fork.stream("s", [{"role": "user", "content": prompt}]))
        results[prompt] = fork.last_usage.input_tokens

    threads = [threading.Thread(target=run, args=(p,)) for p in ("a", "b") * 20]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {"a": 100, "b": 200}


def test_a_retrying_client_forks_its_inner_client_and_starts_its_own_retry_count():
    inner = ollama_with_usage({"q": 3})
    wrapper = RetryingClient(inner)
    wrapper.retries = 5
    fork = wrapper.fork()
    assert fork is not wrapper and fork._inner is not inner and fork.retries == 0
    list(fork.stream("s", MESSAGES))
    assert fork.last_usage == Usage(3, 1) and wrapper.last_usage is None


def test_a_client_without_fork_is_wrapped_as_it_is():
    class Plain:
        model = "plain"
        last_usage = None

    plain = Plain()
    assert RetryingClient(plain).fork()._inner is plain


def test_a_fake_is_shared_by_its_forks():
    fake = FakeLLM(["hello"])
    assert fake.fork() is fake
