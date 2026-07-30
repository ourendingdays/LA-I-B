## Chat Assistant

Chat Assistant is designed to integrate and implement smple RAG and Conversational Bot. It simulates a real-world scenario where we talk to LLM. asking questions based on content from loaded .txt/.pdf documents or just simply so.

> Bo is build with LangChain and HugginFace 

### Worflow and Steps

1. Loading document using LangChain for different sources
 - using `TextLoader`, `PyMuPDFLoader` for fiels and `WebBaseLoader` for websites
2. Splitting long documents using text splitters into chunks with `RecursiveCharacterTextSplitter`
3. Generating embeddings using `HuggingFaceEmbeddings` embedding models
4. Storing embeddings using Chroma's vector database
5. Retrieving similar documents with standard similarity search
