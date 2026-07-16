# Notebooks

Every notebook has been build and tested with `CPX22 — 2 vCPUs, 4 GB RAM`. The notebooks are made to be lighweight, fast and eccessible, which means :

- anyone can run the code, even on the not the strongest hardware
- no model is beeing downloaded, as is often the case with hugging face's `transformer`
- when there is a need of a model from HF, the hosted inference API is used (`InferenceClient`) - must have <b>HuggingFace token</b>


## Essentials

1. install requirements.txt
``` bash
cd ra-i-g/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Jupyter Notebook Kernel

3. Create HuggingFace Token at `huggingface.co/settings/tokens`


## Notebook Descriptions

- document_analysis_llm.ipynb 
This notebook extracts and summarizes text from the .PDF-file. It then generates questions using LLM , to the extracted text and subsequently answers them with the same model.

<b>Build with</b>
- pdfplumber                    : Extracing text rom the PDF
- `transformers` (Hugging Face) : For the powerfull  powerful, free LLM.
- nltk                          : Text-procesing NLP library

- rag_langchain.ipynb
This notebooks takes text data (.txt), splits into chnks, makes embeddings out of them and sues FAISS vectro store. Then, using LLM and given context searche using VectorDB, answers user's questions.

Build with 

- `transformers` (Hugging Face) : For the powerfull  powerful, free LLM.
- `sentence-transformers`       : The easiest way to get a top-tier embedding model
- `faiss-cpu `                  : Facebook AI’s blazing-fast, free vector search library; vector store
- `langchain`                   : Only using its text splitter, which is a smart shortcut that saves hours of regex pain.
- `duckduckgo`                  : Search Engine
- `chroma`                      : Vector DB 
