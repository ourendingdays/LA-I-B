# Must insert the parent directory of the frontend folder into sys.path to allow imports from src
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import psutil
import pytz
import streamlit as st


st.set_page_config(page_title="LA(I)B", page_icon=":material/neurology:", layout="wide", initial_sidebar_state="expanded")
st.logo(image="frontend/assets/laib-logo-flask.svg",
        icon_image="frontend/assets/laib-icon-flask.svg", size="large")

# Define the pages
page_main           = st.Page("pages/main_page.py", title="Home", icon=":material/home:")
page_doc_analysis   = st.Page("pages/rag.py", title="RAG", icon=":material/document_search:")
page_web_agent      = st.Page("pages/web_agent.py", title="AI Agents", icon=":material/robot_2:")
page_chat_bot       = st.Page("pages/chat_bot.py", title="Chat Bot", icon=":material/chat:")

# Set up navigation
pg = st.navigation({
    "Overview": [page_main],
    "Natural Language Processing": [page_doc_analysis, page_web_agent, page_chat_bot],
})

# Run the selected page
pg.run()

with st.sidebar:
    st.divider()
    with st.container(horizontal=True):
        st.caption("@ Pavlo Mospan. 2026")
        st.badge("v0.1.1", color="green")

    ram = psutil.virtual_memory()           # RAM usage
    disk_usage = psutil.disk_usage('/')     # Disk usage for a specific path

    st.caption(f"Compute · RAM {ram.percent:.0f}%  · Disk Usage {disk_usage.percent:.0f}%")
    st.caption(f"CET Time: {datetime.now(pytz.timezone('CET')).strftime('%Y-%m-%d %H:%M')}")


    if st.button("yo?", type="secondary"):
        st.write("yo")