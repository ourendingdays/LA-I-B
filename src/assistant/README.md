## Chat Assistant

Chat Assistant is designed to integrate and implement simple RAG and Conversational Bot. It simulates a real-world scenario where we talk to LLM, asking questions based on content from loaded .txt/.pdf documents or just simply so.

> Bot is build with LangChain and HugginFace, using code from the notebooks/langchain

### Worflow and Steps

1. Loading document using LangChain for different sources
 - using `TextLoader`, `PyMuPDFLoader` for fiels and `WebBaseLoader` for websites
2. Splitting long documents using text splitters into chunks with `RecursiveCharacterTextSplitter`
3. Generating embeddings using `HuggingFaceEmbeddings` embedding models
4. Storing embeddings using Chroma's vector database
5. Retrieving similar documents with standard similarity search

he Biggest difference between this Assistant and other `pipelines` is in document loading : <b>langchain based</b> functions are cnverting text into Docuemnts from the et go, so there is no need to transform them into Docuemnts after splitting, as other `pipelines` chunk loading and splitting functions do ; we can just use `RecursiveCharacterTextSplitter` and use its output directly.