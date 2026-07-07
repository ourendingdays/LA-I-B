import streamlit as st
import psutil


st.set_page_config(
    page_title="LA(I)B",
    page_icon=":material/neurology:",
    layout="wide",
)

st.logo("frontend/assets/app-icon-C-monogram-1024.png", size="large")

# Define the pages
page_main = st.Page("pages/main_page.py", title="Home", icon=":material/home:")
page_doc_analysis = st.Page("pages/document_analysis.py", title="Document Analysis", icon=":material/document_search:")

# Set up navigation
pg = st.navigation({
    "Overview": [page_main],
    "Natural Language Processing": [page_doc_analysis],
})

# Run the selected page
pg.run()

with st.sidebar:
    st.divider()
    st.caption("@ Pavlo Mospan. 2026")
    
    ram = psutil.virtual_memory()           # RAM usage
    disk_usage = psutil.disk_usage('/')     # Disk usage for a specific path

    st.caption(f"Hetzner CPX22 · RAM {ram.percent:.0f}%  · Disk Usage {disk_usage.percent:.0f}%")