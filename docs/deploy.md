# Running and deploying Cairn

## On your machine

```bash
pip install -e ".[serve,llm]"                      # add ",dense" for hybrid search with the embedding model
python scripts/fetch_corpus.py && python scripts/ingest.py     # BM25 needs only this
python scripts/embed_chunks.py                                  # only for hybrid search
python scripts/serve.py --llm none --retriever bm25 --no-auth   # http://127.0.0.1:8000
```

`--llm none` shows the retrieved passages and no generated answer; `--llm ollama --model gemma4:e4b`
(Ollama running) answers in full; `--llm anthropic` needs `ANTHROPIC_API_KEY` and `pip install -e ".[llm]"`.
Without `--no-auth` the server needs `CAIRN_API_KEY` (16 or more characters) in its environment.

## With Docker

```bash
docker compose up                       # retrieval only, http://localhost:8000
docker compose --profile llm up         # also a local model through Ollama, on http://localhost:8001
docker compose exec ollama ollama pull gemma4:e4b      # once, for the second one
```

The image downloads the corpus at the pinned tag and builds the indexes while it is built, so the
first build needs the network and takes a few minutes. `--build-arg WITH_DENSE=1` adds the embedding
model (torch, CPU build): a 2.65 GB image against 457 MB, and 450 to 750 MiB of memory at run time
against about 93 MiB for BM25 only (measured with `docker stats` on one machine, after a few requests).

## The public demo: what it is and what it is not

The demo is **retrieval only**: it shows the passages that match a question, with file, lines and a
viewer, and says plainly that no language model is attached. This is a choice made to cost nothing:
a free host cannot run a model like `gemma4:e4b` (it needed about 40 s per answer on a 12-core
laptop), and a paid API key would turn every visitor into a bill. Running it locally with Ollama gives
full answers with the same page.

The page and the API are open (no key) and protected by the rate limit and by there being nothing to
spend. Behind a hosting proxy all visitors share the proxy's address, so the limit is shared too
(10 requests per second, bursts of 30), which is a global brake, not a fair share per visitor.

## Hugging Face Spaces (what `scripts/make_space.py` prepares)

A Space on the Docker SDK builds this `Dockerfile` (its last stage, the demo) and runs it on port 8000.

```bash
python scripts/make_space.py --out ../cairn-space              # BM25 only; add --dense for hybrid
cd ../cairn-space
git init -b main && git add . && git commit -m "Cairn demo"
git remote add space https://huggingface.co/spaces/<your-user>/cairn
git push space main            # asks for your username and a write token from your settings
```

Create the Space first on huggingface.co (New Space, SDK: Docker, hardware: CPU basic), and create
the token yourself; neither is something to hand to anyone else. The build clones the FastAPI
repository and chunks it, so it takes a few minutes.

**Check the free tier before relying on it.** The Hub documentation lists CPU basic as 2 vCPU and
16 GB at no cost, and says free Spaces go to sleep after a period of inactivity (48 hours for the
free CPU hardware, per the Python library documentation). Forum threads from 2026 report new or
free accounts getting a `cpu-basic quota limit` error (403), and one staff reply saying a PRO
subscription is needed; the documentation and the behaviour did not agree when this was checked, and
free tiers change. Sources: [Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview),
[a quota thread](https://discuss.huggingface.co/t/free-cpu-basic-quota-is-0-on-new-account/180764),
[another](https://discuss.huggingface.co/t/can-hugging-face-pro-run-a-docker-space-on-the-free-cpu-basic-tier/177957).
If a Space cannot be created for free, the BM25-only image needs about 93 MiB of memory and runs on
any host that runs a Docker image; `docker build -t cairn-demo .` is the whole build.

## What was not done

- **The demo is not deployed by me**: it needs your account and token. Everything up to the `git push` to the Space is prepared and tested locally.
- No TLS or custom domain of its own (the host provides them), no per-user keys, no monitoring beyond `/metrics`.
