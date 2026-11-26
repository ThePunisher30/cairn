"""The LlamaIndex pipelines as a user would write them (Phase 11, for the lines-of-code count).

`bench/frameworks/llamaindex_build.py` does the same work with the timing, configuration and
statistics the comparison needs; this file is only what you have to write. Two pipelines:

    default_pipeline    read the files, split with the defaults, embed, FAISS, vector search
    selected_pipeline   Markdown + code splitting, vector search plus BM25 fused by rank

Both take the list of corpus files (Cairn's `ingest/documents.py` decides which files are the
corpus; here the list is simply passed in) and return a retriever with `.retrieve(query)`.
"""

import faiss
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore

MODEL = "BAAI/bge-small-en-v1.5"
INSTRUCTION = "Represent this sentence for searching relevant passages: "


def default_pipeline(files):
    documents = SimpleDirectoryReader(input_files=files).load_data()
    embed_model = HuggingFaceEmbedding(model_name=MODEL, query_instruction=INSTRUCTION)
    store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(384))
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=embed_model,
    )
    return index.as_retriever(similarity_top_k=10)


def selected_pipeline(files):
    documents = SimpleDirectoryReader(input_files=files).load_data()
    markdown = [d for d in documents if d.metadata["file_name"].endswith(".md")]
    code = [d for d in documents if d.metadata["file_name"].endswith(".py")]
    nodes = MarkdownNodeParser().get_nodes_from_documents(markdown)
    nodes += CodeSplitter(language="python", chunk_lines=30, chunk_lines_overlap=5,
                          max_chars=1500).get_nodes_from_documents(code)  # fmt: skip
    embed_model = HuggingFaceEmbedding(model_name=MODEL, query_instruction=INSTRUCTION)
    store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(384))
    index = VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=embed_model,
    )
    vector = index.as_retriever(similarity_top_k=10)
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    return QueryFusionRetriever(
        [vector, bm25], similarity_top_k=10, num_queries=1, mode="reciprocal_rerank",
        use_async=False, llm=MockLLM(),
    )  # fmt: skip


def smoke(files):
    """python -c "import llamaindex_minimal as m; m.smoke(['a.md', 'b.py'])" """
    for build in (default_pipeline, selected_pipeline):
        print(
            build.__name__,
            [n.node.text[:40] for n in build(files).retrieve("how to enable CORS")[:2]],
        )
