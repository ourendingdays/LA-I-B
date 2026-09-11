### Run the code as a module, not a script

From the project root: 

```bash
python -m src.rag.pipelines.simple_doc_analyzer

python -m src.rag.pipelines.simple_rag

python -m src.rag.pipelines.vector_search_rag

python -m src.rag.pipelines.simple_graphrag

python -m src.assistant.chat_assistant

python -m src.rag.agents.web_search

python -m src.rag_visual.src.keyword_search search "Great"
# python -m src.rag_visual.src.keyword_search tf 424 "bear"
# python -m src.rag_visual.src.keyword_search idf grizzly
# python -m src.rag_visual.src.keyword_search tfidf 424 push
```

Or: 

```bash
python src/rag/hugging_face_client
```