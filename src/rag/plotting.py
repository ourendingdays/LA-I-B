# Data Science Libraries
import plotly.graph_objects as go


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