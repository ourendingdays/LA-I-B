# Custom Modules
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.agents.web_search import WebSearchAgent
from src.rag.agents.research_agent import WebResearchAgent
from src.rag.config import load_config
from src.rag.visualization.charts_and_plots import create_distance_bar_chart

# Standard Libraries
import streamlit as st


# ------------ Streamlit UI ------------
st.markdown("# Chat Bot :material/chat:")
st.sidebar.markdown("##### Gentle Chat Bot :material/chat:")