# RAG Overview

Colelction of several Retrieval-Augmented Generation (RAG) and Agent-based approaches.

```
rag/
├── agents/
├── clients/
├── configs/
├── documents/
├── pipelines/
├── storage/
├── visualization/
├── config.py
└── README.md
```

---

## `config.py`

Loads YAML configuration files into plain dicts.

- `load_config(file_path)` — reads a YAML file (e.g. `configs/rag_simple.yaml`, `configs/vector_store.yaml`) and returns its contents as a `dict`.

---

## `configs/`

YAML configuration files, not code.

- `rag_simple.yaml`     — Configuration for the models used for RAG : model settings (prompt template, candidate LLMs, summarization model), text-splitter defaults, embedding model name, top-k.
- `vector_store.yaml`   — Configuration for ChromaDB: where documents live on disk, where the Chroma database persists, and the default collection name.

---

## `clients/`

Thin wrappers around external services/models

### `hugging_face_client.py` — `HuggingFaceClient`
Base class for every RAG/agent class in this package. Wraps the Hugging Face `InferenceClient` for **remote** LLM calls.

- `ask_model(query, prompt, context, model, max_tokens)` — sends a system+user chat completion request, returns the model's text response.
- `get_working_models(models)` / `test_model(model_name)` — HF's free-tier models rotate availability; this checks which candidates currently respond before picking one.
- `request_model(...)` — fallback for the raw HF inference REST endpoint (used for pipeline tasks that don't fit `chat_completion`, e.g. summarization via the classic inference API).

Every RAG/agent class (`SimpleRAG`, `VectorSearchRAG`, `AgenticRAG`, `GraphRAG`, `SimpleDocumentAnalyzer`) inherits from this, cause they all need LLM calls and model availability checks.

### `embedding_client.py` — `EmbeddingClient`
Owns the **local** sentence-transformer embedding model (`all-MiniLM-L6-v2` by default) for vector retrieval. Bridges two different consumers:

- `get_langchain_embeddings()` — returns a LangChain-compatible `HuggingFaceEmbeddings` object, for `Chroma.from_documents(embedding=...)`.
- `encode(texts)` — returns raw `numpy` vectors via `SentenceTransformer`, for FAISS (which doesn't use LangChain's embeddings interface at all).

---

## `documents/`

Turns files into text, and text into chunks. Purely functional — no state, no classes; every RAG variant imports from here identically.

### `loaders.py`
- `load_document(file_path)` — reads a single `.txt` or `.pdf` file and returns its full extracted text as a `str`.
- `load_documents(folder_path)` — loads every `.txt`/`.pdf` in a folder and returns a `List[str]` (one entry per file).

### `splitters.py`
- `split_text_into_chunks(text, ts_chunk_size, ts_chunk_overlap, source=None)`
  — fixed-size chunking via LangChain's `RecursiveCharacterTextSplitter`.
  Returns **both** a plain `List[str]` (for FAISS) and a `List[Document]` wrapped with `{"source": ...}` metadata (for Chroma) — same split, two shapes, so callers don't have to convert between them.
- `split_text_into_passages(text, word_limit=200)` — sentence-aware chunking via NLTK (`sent_tokenize`), combining whole sentences into ~200-word passages. Used  for instance, by `SimpleDocumentAnalyzer` for question generation, where chunk boundaries mid-sentence would hurt quality.
- `_ensure_punkt_tab()` — checks locally before calling `nltk.download()` to save time, so the NLTK data check only happens (and only costs a network round-trip) the first time, and only for code paths that actually call `split_text_into_passages`.

---

## `storage/

Read about the difference between Chroma and Faiss [here](../../notebooks/rag/README.md)

### chroma_store.py` — `ChromaStore`

Lifecycle of Chroma collections -  it only ever receives already-chunked `Document` objects.

- `__init__(embedding_client, persist_directory=None)` — `persist_directory=None`
    - gives an ephemeral, in-memory store (one-off session, e.g. `AgenticRAG`); 
    - a real path persists to disk and survives restarts (e.g. `VectorSearchRAG`).
- `get_or_create(chunks, collection_name)`
    - Returns a Chroma vector store instance based on the provided collection name and chunks.
- `list_collections(chunks, collection_name)` 
    - Returns collections already present.
- `generate_collection_name(prefix)` 
    - arbitrary unique collection name (`prefix_<uuid8>`), so that  each session/upload can get its own isolated collection without needing a separate Chroma client per collection.

> One Chroma client/persist directory can hold many independently-named collections — a single `ChromaStore` instance is shared and reused, not recreated per document.

## `faiss_index.py` — `FaissIndex`

- `create_embeddings_and_index(chunks, model_name)` — builds a FAISS `IndexFlatL2` from chunk embeddings (via `EmbeddingClient.encode`).
- `retrieve_chunks(query, top_k)` — embeds the query, searches the index, returns the nearest chunks + their distances.
- `get_embedding_visualization_data()` — exposes chunk/query embeddings and retrieved indices for `create_embedding_scatter`.

---

## `visualization/charts_and_plots.py`

Charts and plots

- `create_distance_bar_chart(chunks, distances)` — horizontal bar chart of retrieved chunks sorted by relevance (lowest distance = most relevant, on top). Returns `(fig, order)`.
- `create_embedding_scatter(chunk_embeddings, query_embedding, chunks, retrieved_indices)` -  2D PCA projection of the chunk embedding space, highlighting which chunks were retrieved for the current query and where the query sits relative to them.
- `create_graph_visualization(kg, highlight_nodes, highlight_edges)` — renders a `networkx` knowledge graph as an interactive `pyvis` HTML graph, optionally highlighting a traversal path (used by `GraphRAG` to show *why* it answered the way it did).

---

## `pipelines/`


### Run the code as a module, not a script

Example from the project root: 

```bash
python -m src.rag.pipelines.simple_rag
```

Or:  

```bash
python src/rag/hugging_face_client
```


> The actual end-to-end RAG and Agents implementations. Each composes `clients/`, `documents/`, and `storage/` — none of them reimplement loading, chunking, or storage logic themselves.


#TODO: check for consistency ;ter, sicne i am redesigning code a lot as of late.

### `simple_doc_analyzer.py` — `SimpleDocumentAnalyzer`
Document summarization. Also, creates quesions to the extracted passages and answers these qestions based on the context, provided in this passage — no retrieval.

- `summarize_document(input_text, model)` — HF summarization pipeline.
- `generate_questions(passage, min_questions, model)` — LLM-generated questions about a passage, with a sentence-chunk fallback if the model returns too few.
- `generate_passages_with_questions(input_text, model)` — splits text into passages (`split_text_into_passages`), generates and answers questions per passage. Returns `(passages, {idx: {question: answer}})`.


### `simple_rag.py` — `SimpleRAG`
The Simplest "Ask a question based on the document's context" RAG.
Single-document, ephemeral, FAISS-backed retrieval. Rebuilds its index every call — no persistence.
- `preprocess_document(file_path, query, ts_chunk_size, ts_chunk_overlap, embeddings_top_k)` — loads, chunks, embeds, indexes, and retrieves in one call. Returns `(retrieved_chunks, distances)`.

### `vector_search_rag.py` — `VectorSearchRAG`

Multi-document, persisted, Chroma-backed retrieval. Doesn't build or own a
vector store itself — it's handed a `ChromaStore` and a `Chroma` instance to
query. Meant for "search across everything I've uploaded, across sessions."

- `query_rag_system(query_text, vector_store, prompt, model, top_k, max_tokens)` — retrieves top-k chunks from the given `Chroma` store, builds context, asks the LLM.
- `initialize_knowledge_base(folder_path, collection_name, ts_chunk_size, ts_chunk_overlap)` - Initializes the knowledge base by loading documents from a folder, splitting them into chunks, and creating a vector store.


### `simple_graphrag.py` — `GraphRAG`
*(or `graph/graph_rag.py`)*

Knowledge-graph-based retrieval via multi-hop traversal — answers questions by walking entity relationships instead of nearest-neighbor vector search. 

- `build_knowledge_graph(text, model)` — extracts (head, relation, tail) triples via the LLM, builds a `networkx.DiGraph`, stores it on `self.KG`.
- `retrieve_graph_context(entity, max_depth)` — DFS traversal from a starting entity, returns the context string plus the visited nodes/edges (for highlighting in `create_graph_visualization`).
- `extract_entity_from_question(question, model)` — LLM picks which known graph node a free-text question is about, so the user doesn't have to name the entity manually.
- `graph_rag_answer(question, model, entity=None, max_depth)` — full pipeline: auto-extract entity if not given → traverse → answer.

---

## `agents/`

Decision-making layers built on top of the pipelines above — they decide
*whether* and *how* to retrieve, not just how to answer.

### `agentic_rag.py` — `AgenticRAG`
Routes each query through an LLM-based decision (search vs. answer directly)
before touching retrieval at all — avoids always searching regardless of
whether the question needs it.

- `agent_controller(query, model)` — LLM call that returns `"search"` or `"direct"`.
- `build_vector_store(folder_path, ts_chunk_size, ts_chunk_overlap)` — loads and chunks a folder, then delegates the actual Chroma build to
  `ChromaStore.create_vector_store` (own collection, ephemeral by default, named via `ChromaStore.generate_collection_name`).
- `answer(query, model, top_k, max_tokens)` — routes the query, retrieves + answers if `"search"`, otherwise answers directly from the LLM's general knowledge. Returns `{"answer": str, "action": "search"|"direct"}`.

### `web_search.py` — `WebSearchAgent`
Answers questions using live web search (DuckDuckGo) instead of any local
document store — independent of the rest of the RAG pipeline, useful for
questions no uploaded document could answer.

- `web_search(query, max_results)` — DuckDuckGo text search, formatted as title/body context.
- `change_model(new_model)` - can change the model

### `web_research_agent.py` — `WebResearchAgent`
Live-web research agent — searches the web, fetches and reads the actual pages (not just search snippets), chunks them, and ranks passages by semantic similarity to the query before summarizing. A deeper alternative to `WebSearchAgent`, which only uses raw DuckDuckGo result snippets.

- `search_web(query, max_results)` / `unwrap_ddg(url)` — DuckDuckGo search, cleaning DDG's redirect-wrapped URLs.
- `fetch_text(url, timeout)` — downloads and parses a page with BeautifulSoup, stripping nav/script/footer noise, with a meta-description fallback if `<p>` tags yield nothing usable.
- `run(query, config)` — full pipeline: search → fetch → chunk (`documents/splitters.chunk_passages`) → embed (`EmbeddingClient.encode`) → rank by cosine similarity → summarize. Returns `{"query", "passages", "summary", "time"}`.
- `_extractive_summary(...)` — no-LLM summary mode: ranks *sentences* within the top passages by similarity to the query and stitches the best ones together verbatim, with source attribution — zero hallucination risk since nothing is generated, only selected.

---

## A typical Workflow

```
load_document → split_text_into_chunks → EmbeddingClient.encode + FAISS index / ChromaStore.get_or_create → retrieve and ask model
```