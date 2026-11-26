"""The LlamaIndex + FAISS rows of the comparison (Phase 11).

Everything LlamaIndex-specific is in this file. It returns a `retrieve(query, k)` function that
yields `RetrievedNode`s, which `cairn.evaluation.adapters.CharSpanRetriever` turns into hits.

The configuration is a dict (see `rows.py`):
    chunking          {"kind": "sentence" | "md+sentence" | "md+code", ...sizes}
    exclude_metadata  keep file name, size and dates out of the text that is embedded
    query_instruction use bge's recommended query instruction
    retriever         "vector" | "bm25" | "fusion"
    faiss             "flat_l2" | "flat_ip" | "hnsw"
"""

from pathlib import Path

from common import BGE, BGE_QUERY_INSTRUCTION, K, Stopwatch

from cairn.evaluation.adapters import RetrievedNode
from cairn.ingest.documents import load_documents


def _documents(root, limit_files):
    from llama_index.core import SimpleDirectoryReader

    listing = load_documents(root)  # the same files, by the same rule, as every Cairn row
    if limit_files:
        listing = listing[:limit_files]
    reader = SimpleDirectoryReader(input_files=[str(Path(root) / d.path) for d in listing])
    documents = reader.load_data()
    root = Path(root).resolve()
    path_of = {}
    for doc in documents:
        rel = Path(doc.metadata["file_path"]).resolve().relative_to(root).as_posix()
        path_of[doc.doc_id] = rel
    return documents, path_of, len(listing)


def _parse(documents, path_of, chunking, exclude_metadata):
    from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser, SentenceSplitter

    keep = not exclude_metadata
    kind = chunking["kind"]
    if kind == "sentence":
        splitter = SentenceSplitter(
            chunk_size=chunking["chunk_size"],
            chunk_overlap=chunking["chunk_overlap"],
            include_metadata=keep,
        )
        return splitter.get_nodes_from_documents(documents)
    markdown = [d for d in documents if path_of[d.doc_id].endswith(".md")]
    code = [d for d in documents if not path_of[d.doc_id].endswith(".md")]
    nodes = MarkdownNodeParser(include_metadata=keep).get_nodes_from_documents(markdown)
    if kind == "md+sentence":
        splitter = SentenceSplitter(
            chunk_size=chunking["chunk_size"],
            chunk_overlap=chunking["chunk_overlap"],
            include_metadata=keep,
        )
        nodes = splitter.get_nodes_from_documents(nodes)  # cut sections that are still too long
        nodes += splitter.get_nodes_from_documents(code)
    elif kind == "md+code":
        code_splitter = CodeSplitter(
            language="python",
            chunk_lines=chunking["chunk_lines"],
            chunk_lines_overlap=chunking["chunk_lines_overlap"],
            max_chars=chunking["max_chars"],
        )
        nodes += code_splitter.get_nodes_from_documents(code)
    else:
        raise ValueError(f"unknown chunking {kind!r}")
    return nodes


def _faiss_index(kind, dim):
    import faiss

    if kind == "flat_l2":
        return faiss.IndexFlatL2(dim)  # what the LlamaIndex documentation uses
    if kind == "flat_ip":
        return faiss.IndexFlatIP(dim)
    if kind == "hnsw":
        return faiss.IndexHNSWFlat(dim, 32)  # FAISS defaults otherwise (efSearch 16)
    raise ValueError(f"unknown faiss index {kind!r}")


def build_llamaindex(cfg, root, limit_files=None):
    """Returns (retrieve, texts, info). `texts` maps each path to the text LlamaIndex read."""
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.schema import MetadataMode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore

    info = {}
    with Stopwatch() as total:
        with Stopwatch() as reading:
            documents, path_of, n_files = _documents(root, limit_files)
        if cfg.get("normalize_newlines"):
            for doc in documents:
                doc.set_content(doc.text.replace("\r\n", "\n"))
        if cfg["exclude_metadata"]:
            for doc in documents:
                doc.excluded_embed_metadata_keys = list(doc.metadata)
                doc.excluded_llm_metadata_keys = list(doc.metadata)
        texts = {path_of[d.doc_id]: d.text for d in documents}
        with Stopwatch() as parsing:
            nodes = _parse(documents, path_of, cfg["chunking"], cfg["exclude_metadata"])
        vector = None
        indexing = Stopwatch()
        indexing.seconds = 0.0
        if cfg["retriever"] != "bm25":  # a BM25-only row embeds nothing
            embed_model = HuggingFaceEmbedding(
                model_name=BGE,
                query_instruction=BGE_QUERY_INSTRUCTION if cfg["query_instruction"] else None,
                device="cpu",
            )
            with Stopwatch() as indexing:
                store = FaissVectorStore(faiss_index=_faiss_index(cfg["faiss"], 384))
                index = VectorStoreIndex(
                    nodes,
                    storage_context=StorageContext.from_defaults(vector_store=store),
                    embed_model=embed_model,
                    show_progress=False,
                )
            vector = index.as_retriever(similarity_top_k=K)
        retriever = vector
        if cfg["retriever"] in ("bm25", "fusion"):
            from llama_index.retrievers.bm25 import BM25Retriever

            with Stopwatch() as bm25_build:
                bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=K)
            info["bm25_seconds"] = bm25_build.seconds
            retriever = bm25
            if cfg["retriever"] == "fusion":
                from llama_index.core.llms import MockLLM
                from llama_index.core.retrievers import QueryFusionRetriever

                retriever = QueryFusionRetriever(
                    [vector, bm25],
                    similarity_top_k=K,
                    num_queries=1,  # the default (4) asks a language model to rewrite the query
                    mode="reciprocal_rerank",
                    use_async=False,
                    verbose=False,
                    llm=MockLLM(),  # required even when num_queries=1 never calls it (else OpenAI)
                )
    info.update(
        build_seconds=total.seconds,
        read_seconds=reading.seconds,
        parse_seconds=parsing.seconds,
        embed_index_seconds=indexing.seconds,
        files=n_files,
        documents=len(documents),
        chunks=len(nodes),
        embed_texts=[n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes],
        embedder=None,
    )

    def retrieve(query, k):
        found = retriever.retrieve(query)
        out = []
        for item in found[:k]:
            node = item.node
            out.append(
                RetrievedNode(
                    path_of[node.ref_doc_id],
                    node.start_char_idx,
                    node.end_char_idx,
                    float(item.score or 0.0),
                    node.get_content(metadata_mode=MetadataMode.NONE),
                )
            )
        return out

    return retrieve, texts, info
