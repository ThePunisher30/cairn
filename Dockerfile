# syntax=docker/dockerfile:1
#
# Three stages:
#   data     downloads the pinned corpus (FastAPI 0.143.0), chunks it and builds the indexes
#   runtime  the service and its data (no git, no build tools); `--target runtime`
#   demo     the runtime plus the safe public settings: no model, no key, a rate limit (the LAST
#            stage, so a plain `docker build .` and a Hugging Face Space build this one)
#
#   docker build -t cairn-demo .                                   BM25 only, about 0.3 GB
#   docker build --build-arg WITH_DENSE=1 -t cairn-demo .          BM25 + embeddings (torch, CPU)
#   docker build --target runtime -t cairn .                       what docker-compose.yml uses
#
# WITH_DENSE=1 adds torch (the CPU build, installed explicitly) and the embedding model: a much
# bigger image (2.65 GB measured, against 457 MB) and 450 to 750 MiB of memory at run time
# (measured), against about 93 MiB for BM25 only.

ARG PYTHON_VERSION=3.12

# ---- the data ---------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS data
ARG WITH_DENSE=0
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CAIRN_DATA_DIR=/data HF_HOME=/data/hf
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml ./
COPY src ./src
RUN if [ "$WITH_DENSE" = "1" ]; then \
      pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[dense]"; \
    else \
      pip install .; \
    fi
COPY scripts ./scripts
# the clone is checked out at the tag in CAIRN_CORPUS_TAG (default 0.143.0); only the three
# folders the corpus is made of are kept
RUN python scripts/fetch_corpus.py \
    && python scripts/ingest.py \
    && if [ "$WITH_DENSE" = "1" ]; then python scripts/embed_chunks.py; fi \
    && mkdir -p /out/raw/fastapi \
    && cp -r /data/raw/fastapi/docs /data/raw/fastapi/fastapi /data/raw/fastapi/docs_src \
       /data/raw/fastapi/LICENSE /out/raw/fastapi/ \
    && cp /data/chunks.jsonl /out/ \
    && if [ "$WITH_DENSE" = "1" ]; then cp /data/dense-*.npz /out/ && cp -r /data/hf /out/hf; fi

# ---- the service ------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime
ARG WITH_DENSE=0
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    CAIRN_DATA_DIR=/data HF_HOME=/data/hf
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# an editable install keeps cairn.config.PROJECT_ROOT at /app, where eval/ and scripts/ are
RUN if [ "$WITH_DENSE" = "1" ]; then \
      pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[serve,llm,dense]"; \
    else \
      pip install -e ".[serve,llm]"; \
    fi
COPY scripts/serve.py ./scripts/serve.py
COPY eval/llm_prices.json ./eval/llm_prices.json
COPY --from=data /out /data
# which retriever this image can run: the dense index and the embedding model exist only with
# WITH_DENSE=1
RUN if [ "$WITH_DENSE" = "1" ]; then echo hybrid; else echo bm25; fi > /app/retriever \
    && useradd --create-home --uid 10001 cairn \
    && chown -R cairn /data
USER cairn
# with the embedding model baked in, never go to the network for it
ENV HF_HUB_OFFLINE=${WITH_DENSE:+1}
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"
# Any arguments to `docker run` are added after these, so `docker run cairn --llm none --no-auth`
# works (a later --retriever overrides the default); with none given the service needs
# CAIRN_API_KEY and a model (see docker-compose.yml).
ENTRYPOINT ["sh", "-c", "exec python scripts/serve.py --host 0.0.0.0 --port 8000 --retriever \"$(cat /app/retriever)\" \"$@\"", "cairn"]

# ---- the public demo (the last stage: the default target) ----------------------------------
FROM runtime AS demo
# No model is attached and there is no key, so the page and the API are open: the retrieval-only
# mode (nothing to spend) and the rate limit are the protection. Behind a hosting proxy every
# visitor arrives from the proxy's address, so this limit is in effect shared by all visitors
# (10 requests per second, bursts of 30); trusting X-Forwarded-For instead would let a visitor
# choose their own address, so it is not done.
CMD ["--llm", "none", "--no-auth", "--rate", "10", "--burst", "30", "--log-level", "INFO"]
