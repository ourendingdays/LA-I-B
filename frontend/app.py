# Must insert the parent directory of the frontend folder into sys.path to allow imports from src
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import psutil


st.set_page_config(page_title="LA(I)B", page_icon=":material/neurology:", layout="wide", initial_sidebar_state="expanded")
st.logo("frontend/assets/app-icon-C-monogram-1024.png", size="large")

# Define the pages
page_main = st.Page("pages/main_page.py", title="Home", icon=":material/home:")
page_doc_analysis = st.Page("pages/rag.py", title="RAG", icon=":material/document_search:")
page_ai_agent = st.Page("pages/ai_agent.py", title="AI Agent", icon=":material/robot_2:")

# Set up navigation
pg = st.navigation({
    "Overview": [page_main],
    "Natural Language Processing": [page_doc_analysis, page_ai_agent],
})

# Run the selected page
pg.run()

with st.sidebar:
    st.divider()
    st.caption("@ Pavlo Mospan. 2026")
    
    ram = psutil.virtual_memory()           # RAM usage
    disk_usage = psutil.disk_usage('/')     # Disk usage for a specific path

    st.caption(f"Hetzner CPX22 · RAM {ram.percent:.0f}%  · Disk Usage {disk_usage.percent:.0f}%")