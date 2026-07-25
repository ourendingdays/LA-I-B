# Data Science Libraries
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
from nltk.tokenize import sent_tokenize

# Standard Libraries
from typing import List

def _ensure_punkt_tab():
    """Downloads punkt_tab only if it isn't already present locally — skips the network check entirely otherwise."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

def split_text_into_chunks(knowledge_text: str, ts_chunk_size: int = 150, ts_chunk_overlap: int = 20, source: str = None) -> List[str]:
    """
    Splits the given knowledge text into chunks.
    RecursiveCharacterTextSplitter splits on paragraphs ("\n\n"), then newlines ("\n"), then spaces (" "), to keep semantically related text together as much as possible.

    Args:
        knowledge_text    (str)           : The text to be split into chunks.
        ts_chunk_size     (int, optional) : Size of each chunk. Defaults to 150.
        ts_chunk_overlap  (int, optional) : Overlap between chunks. Defaults to 20.
        source            (str, optional) : Optional filename/source to tag onto each chunk's metadata.

    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ts_chunk_size,
        chunk_overlap=ts_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks_text = text_splitter.split_text(knowledge_text)
    metadata = {"source": source} if source else {}
    chunks_doc = [Document(page_content=c, metadata=metadata) for c in chunks_text]

    return chunks_text, chunks_doc

def split_text_into_passages(knowledge_text: str, word_limit: int = 200) -> List[str]:
    """
    Splits the given knowledge text into passages.
    This code tokenizes the document into sentences and combines them into manageable passages for subsequent steps.

    Args:
        knowledge_text    (str)           : The text to be split into passages.
        word_limit        (int, optional) : Maximum number of words in each passage. Defaults to 200.

    Returns:
        list: List of text passages.
    """
    _ensure_punkt_tab()  # Ensure the punkt_tab tokenizer is available
    sentences = sent_tokenize(knowledge_text)

    # combining sentences into passages
    passages = []
    current_passage = ""
    for sentence in sentences:
        if len(current_passage.split()) + len(sentence.split()) < word_limit:  # adjust the word limit as needed
            current_passage += " " + sentence
        else:
            passages.append(current_passage.strip())
            current_passage = sentence
    if current_passage:
        passages.append(current_passage.strip())
    return passages
