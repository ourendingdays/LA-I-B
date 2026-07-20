# Data Science Libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
nltk.download('punkt_tab')  # not 'punkt'
from nltk.tokenize import sent_tokenize

# Standard Libraries
from typing import List
import os
import yaml
from pathlib import Path
from pypdf import PdfReader
import plotly.graph_objects as go


def load_config(file_path: Path) -> dict:
    """
    Loads configuration from a YAML file.

    Args:
        file_path (Path): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data.
    """
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return data

def load_document(file_path: Path):
    """
    Loads a document (.txt and .pdf formats) from the given file path and splits it into chunks.

    Args:
        file_path         (Path)          : Path to the document file.
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

def split_text_into_chunks(knowledge_text: str, ts_chunk_size: int = 150, ts_chunk_overlap: int = 20) -> List[str]:
    """
    Splits the given knowledge text into chunks.
    RecursiveCharacterTextSplitter splits on paragraphs ("\n\n"), then newlines ("\n"), then spaces (" "), to keep semantically related text together as much as possible.

    Args:
        knowledge_text    (str)           : The text to be split into chunks.
        ts_chunk_size     (int, optional) : Size of each chunk. Defaults to 150.
        ts_chunk_overlap  (int, optional) : Overlap between chunks. Defaults to 20.

    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ts_chunk_size,
        chunk_overlap=ts_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(knowledge_text)
    # print(f"Total number of chunks created: {len(chunks)}")    
    return chunks

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

def create_distance_bar_chart(chunks: list[str], distances: list[float]) -> go.Figure:
    """
    Builds a horizontal bar chart of retrieved chunks sorted by distance
    (lowest = most relevant, shown at top).

    Args:
        chunks (list[str]): List of text chunks.
        distances (list[float]): List of distances corresponding to the chunks.

    Returns:
        tuple[go.Figure, list[int]]: A tuple containing the Plotly Figure object representing the bar chart and the order of the chunks.
    """
    order = sorted(range(len(distances)), key=lambda i: distances[i])
    sorted_chunks = [chunks[i] for i in order]
    sorted_distances = [distances[i] for i in order]
    labels = [f"Chunk {i + 1}" for i in order]
    hover_texts = [c[:300] + ("..." if len(c) > 300 else "") for c in sorted_chunks]

    fig = go.Figure(
        go.Bar(
            x=sorted_distances,
            y=labels,
            orientation="h",
            text=[f"{d:.3f}" for d in sorted_distances],
            textposition="outside",
            hovertext=hover_texts,
            hoverinfo="text",
            marker=dict(
                color=sorted_distances,
                colorscale="Blues_r",
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Distance (lower = more relevant)",
        yaxis_title="Chunk",
        yaxis=dict(autorange="reversed"),
        height=80 + 40 * len(labels),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig, order

