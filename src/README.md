## RAG
### Run the code as a module, not a script

From the project root: 

```bash
python -m src.rag.pipelines.simple_doc_analyzer

python -m src.rag.pipelines.simple_rag

python -m src.rag.pipelines.vector_search_rag

python -m src.rag.pipelines.simple_graphrag

python -m src.assistant.chat_assistant

python -m src.rag.agents.web_search
```

Or: 

```bash
python src/rag/hugging_face_client
```

## Visual RAG 
This is a RAG Search Engine - Movie search CLI

### Keyword Search

##### Setup

```bash
python -m src.rag_visual.cli.keyword_search build   # builds index and saves to cache/. Run once before searching. only rerun if movies.json changes.
```

#### Search

```bash
python -m src.rag_visual.cli.keyword_search bm25search "love story"          # top 5 results
python -m src.rag_visual.cli.keyword_search bm25search "space" --limit 10    # custom limit
python -m src.rag_visual.cli.keyword_search search "Great"                       # basic keyword search
```

#### Scoring Tools

```bash
python -m src.rag_visual.cli.keyword_search tf 4651 "merida"          # term frequency in a doc
python -m src.rag_visual.cli.keyword_search idf "merida"               # how rare a term is
python -m src.rag_visual.cli.keyword_search tfidf 4651 "merida"        # TF × IDF
python -m src.rag_visual.cli.keyword_search bm25idf "grizzly"          # BM25 smoothed IDF
python -m src.rag_visual.cli.keyword_search bm25tf 4651 "merida"       # BM25 TF (k1=1.5, b=0.75)
python -m src.rag_visual.cli.keyword_search bm25tf 4651 "merida" 1.2 0.5  # custom k1 and b
```

**k1** — term saturation (higher = repeated terms matter more, default 1.5)
**b** — length penalty (0 = ignore length, 1 = full penalty, default 0.75)

### Semantic Search

```bash
python -m src.rag_visual.cli.semantic_search verify
python -m src.rag_visual.cli.semantic_search verify_embeddings
python -m src.rag_visual.cli.semantic_search embed_query "funny bear movies"
python -m src.rag_visual.cli.semantic_search search "funny bear movies" --limit 5
python -m src.rag_visual.cli.semantic_search chunk "This is a test text with two chunks" --chunk-size 5 --overlap 2
python -m src.rag_visual.cli.semantic_search semantic_chunk "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence. This is the fifth sentence." --max-chunk-size 3
python -m src.rag_visual.cli.semantic_search embed_chunks
```