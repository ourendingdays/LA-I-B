## Notebook Descriptions

- `document_analysis_with_llm.ipynb` - extracts and summarizes text from the .PDF-file. It then generates questions using LLM , to the extracted text and subsequently answers them with the same model.

    - pdfplumber                    : Extracting text from the PDF
    - `transformers` (Hugging Face) : For the powerfull  powerful, free LLM.
    - nltk                          : Text-procesing NLP library

- `rag_langchain.ipynb`
This notebooks takes text data (.txt), splits into chnks, makes embeddings out of them and sues FAISS vectro store. Then, using LLM and given context searche using VectorDB, answers user's questions.

    - `transformers` (Hugging Face) : For the powerfull  powerful, free LLM.
    - `sentence-transformers`       : The easiest way to get a top-tier embedding model
    - `faiss-cpu `                  : Facebook AI’s blazing-fast, free vector search library; vector store
    - `langchain`                   : Only using its text splitter, which is a smart shortcut that saves hours of regex pain.
    - `duckduckgo`                  : Search Engine
    - `chroma`                      : Vector DB 
