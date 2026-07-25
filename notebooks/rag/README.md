## Notebook Descriptions

- `rag_simple.ipynb`
This notebooks firs of all checks for available models on HF for inference. The takes text data (.txt), splits into chunks, makes embeddings out of them and uses FAISS vector store. Then, using LLM and given context searches for an answer using VectorDB.

    - `InferenceClient`             : HF's Hub Client for the powerfull  powerful, free LLM.
    - `sentence-transformers`       : The easiest way to get a top-tier embedding model
    - `faiss-cpu `                  : Facebook AI’s blazing-fast, free vector search library; vector store
    - `langchain`                   : Only using its text splitter, which is a smart shortcut that saves hours of regex pain.

- `multi_doc_with_vector_search.ipynb`
This takes text data (.pdf), splits into passages of sentences, makes embeddings out of them and stores them in the Chroma vector database. Then, using LLM and given context answers user's questions using VectorDB.

    - `InferenceClient`             : HF's Hub Client for the powerfull  powerful, free LLM.
    - `sentence-transformers`       : The easiest way to get a top-tier embedding model
    - `chroma`                      : Vector DB            
    - `langchain`                   : Only using its text splitter, which is a smart shortcut that saves hours of regex pain.


- `document_analysis_with_llm.ipynb` - extracts and summarizes text from the .PDF-file. It then generates questions using LLM , to the extracted text and subsequently answers them with the same model.

    - pdfplumber                    : Extracting text from the PDF
    - `transformers` (Hugging Face) : For the powerfull  powerful, free LLM.
    - nltk                          : Text-procesing NLP library

- `rag_langchain.ipynb`
This notebooks takes text data (.txt), splits into chunks, makes embeddings out of them and sues FAISS vectro store. Then, using LLM and given context answers user's questions using VectorDB..

    - `transformers` (Hugging Face) : For the powerfull  powerful, free LLM.
    - `sentence-transformers`       : The easiest way to get a top-tier embedding model
    - `faiss-cpu `                  : Facebook AI’s blazing-fast, free vector search library; vector store
    - `langchain`                   : Only using its text splitter, which is a smart shortcut that saves hours of regex pain.
    - `duckduckgo`                  : Search Engine
    - `chroma`                      : Vector DB 

##### FAISS v Chroma
<i>Chroma and FAISS are both implementations of approximate/exact nearest-neighbor search, at different levels of abstraction.</i>

<b>FAISS</b> (Facebook AI Similarity Search) is a low-level library, that has  an index data structure (IndexFlatL2 in code) and does only one thing: given a query vector, find the closest stored vectors. No persistence, no metadata, no collections.

<b>Chroma</b> is a vector database that also does "find nearest vectors", but wrapping it with persistence, metadata storage, named collections, and a query API. Under the hood it uses its own indexing (HNSW via hnswlib).