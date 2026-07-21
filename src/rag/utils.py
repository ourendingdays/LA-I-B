# Data Science Libraries
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
nltk.download('punkt_tab')  # not 'punkt'
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.decomposition import PCA

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
        chunks    (list[str])     : List of text chunks.
        distances (list[float])   : List of distances corresponding to the chunks.

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

def create_embedding_scatter(chunk_embeddings: np.ndarray, query_embedding: np.ndarray, chunks: List[str], retrieved_indices: List[int]) -> go.Figure:
    """
    Projects chunk + query embeddings into 2D via PCA and plots them,
    highlighting which chunks were retrieved for the current query.
    """
    n_samples = chunk_embeddings.shape[0] + 1
    n_components = min(2, n_samples - 1)  # guard against tiny doc/chunk counts

    all_embeddings = np.vstack([chunk_embeddings, query_embedding])
    reduced = PCA(n_components=n_components, random_state=42).fit_transform(all_embeddings)

    # pad to 2D if PCA had to reduce further (e.g. only 1-2 chunks total)
    if reduced.shape[1] < 2:
        reduced = np.hstack([reduced, np.zeros((reduced.shape[0], 1))])

    chunk_coords = reduced[:-1]
    query_coord = reduced[-1]

    retrieved_set = set(retrieved_indices)
    non_retrieved_idx = [i for i in range(len(chunks)) if i not in retrieved_set]
    retrieved_idx = list(retrieved_set)

    def hover_text(i):
        preview = chunks[i][:150] + ("..." if len(chunks[i]) > 150 else "")
        return f"Chunk {i + 1}<br>{preview}"

    fig = go.Figure()

    if non_retrieved_idx:
        fig.add_trace(go.Scatter(
            x=chunk_coords[non_retrieved_idx, 0],
            y=chunk_coords[non_retrieved_idx, 1],
            mode="markers",
            marker=dict(size=8, color="lightgray"),
            text=[hover_text(i) for i in non_retrieved_idx],
            hoverinfo="text",
            name="Other chunks"
        ))

    if retrieved_idx:
        fig.add_trace(go.Scatter(
            x=chunk_coords[retrieved_idx, 0],
            y=chunk_coords[retrieved_idx, 1],
            mode="markers",
            marker=dict(size=13, color="royalblue"),
            text=[hover_text(i) for i in retrieved_idx],
            hoverinfo="text",
            name="Retrieved chunks"
        ))

    fig.add_trace(go.Scatter(
        x=[query_coord[0]],
        y=[query_coord[1]],
        mode="markers",
        marker=dict(size=18, color="crimson", symbol="star"),
        text=["Query"],
        hoverinfo="text",
        name="Query"
    ))

    fig.update_layout(
        xaxis_title="PCA Dimension 1",
        yaxis_title="PCA Dimension 2",
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    return fig