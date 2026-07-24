# Data Science Libraries
import numpy as np
from sklearn.decomposition import PCA

# Standard Libraries
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
from typing import List


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

def create_graph_visualization(kg: "nx.DiGraph", highlight_nodes: set = None, highlight_edges: list = None, height: str = "500px") -> str:
    """
    Renders a networkx DiGraph as an interactive pyvis HTML graph.
    Optionally highlights a traversal path (visited nodes / edges) in a
    distinct color against the rest of the graph.

    Args:
        kg              (nx.DiGraph) : The knowledge graph to render.
        highlight_nodes (set)        : Node names to highlight (e.g. traversal path).
        highlight_edges (list)       : (u, v) tuples to highlight.
        height          (str)        : Pixel height of the rendered graph.

    Returns:
        str: Self-contained HTML for embedding via st.components.v1.html().
    """
    highlight_nodes = highlight_nodes or set()
    highlight_edges = set(highlight_edges or [])

    net = Network(height=height, width="100%", directed=True, notebook=False, bgcolor="#ffffff")
    net.barnes_hut(gravity=-3000, spring_length=150)

    for node in kg.nodes():
        is_highlighted = node in highlight_nodes
        net.add_node(
            node,
            label=node,
            color="#e63946" if is_highlighted else "#a8dadc",
            size=25 if is_highlighted else 15,
            font={"size": 14 if is_highlighted else 10}
        )

    for u, v, data in kg.edges(data=True):
        is_highlighted = (u, v) in highlight_edges
        net.add_edge(
            u, v,
            label=data.get("label", ""),
            color="#e63946" if is_highlighted else "#cccccc",
            width=3 if is_highlighted else 1,
            arrows="to"
        )

    return net.generate_html()