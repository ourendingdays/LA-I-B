# Standard Libraries
import networkx as nx
import os
from pathlib import Path
from pypdf import PdfReader
from typing import List

def load_document(file_path: Path):
    """
    Loads a document (.txt and .pdf formats) from the given file path and splits it into chunks.

    Args:
        file_path (Path): Path to the document file.
    Returns:
        str: Extracted text from the document.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist")

    if file_path.suffix.lower() == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            knowledge_text = f.read()
    elif file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        knowledge_text = ""
        for page in reader.pages:
            knowledge_text += page.extract_text()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Only .txt and .pdf are supported.")

    return knowledge_text

def load_documents(folder_path: Path) -> List[str]:
    """
    Loads all documents (.txt and .pdf formats) from the given folder path.

    Args:
        folder_path (Path): Path to the folder containing document files.
    Returns:
        list: List of extracted texts from the documents.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")

    documents = []
    for file_path in Path(folder_path).glob("*"):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            documents.append(load_document(file_path))
    return documents
