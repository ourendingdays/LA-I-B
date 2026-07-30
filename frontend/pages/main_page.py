import streamlit as st

# Sidebar page marker — keeps the "Gentle Xxx" pattern used across pages
st.sidebar.markdown("##### Gentle Home :material/home:")

# ---------- Hero ----------
col_l, col_r = st.columns([0.75, 0.25], vertical_alignment="bottom")
with col_l:
    st.markdown("# la(i)b")
    st.markdown("###### A collection of Data Science tools, techniques, and models for various AI/ML tasks.")
    st.caption("A user-friendly platform for exploring and analyzing data, building models, and conducting research")
with col_r:
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.badge("v0.1.1", color="green")
        st.badge("Experimental", color="gray")

st.divider()

# ---------- Intro ----------
st.write("The central idea of this project is to introduce, research, implement and integrate the concepts of Data Science and AI in a way that is accessible to everyone, regardless of their technical background and computational capacity of their devices. ")


st.write("")

# ---------- What's inside ----------
st.markdown("##### :material/apps: What's inside")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown("### :material/document_search:")
        st.markdown("**RAG**")
        st.caption("Document analysis")
        st.write(
            "Simple RAG and Graph RAG in one place. Upload a document, "
            "tune chunk size and top-k, watch the embedding space, ask questions."
        )
        st.page_link("pages/rag.py", label="Open RAG", icon=":material/arrow_forward:")

with col2:
    with st.container(border=True):
        st.markdown("### :material/robot_2:")
        st.markdown("**AI Agents**")
        st.caption("Web search & research")
        st.write(
            "A quick DuckDuckGo agent for straight lookups, plus a heavier "
            "research agent that ranks passages semantically before summarizing."
        )
        st.page_link("pages/web_agent.py", label="Open Agents", icon=":material/arrow_forward:")

with col3:
    with st.container(border=True):
        st.markdown("### :material/chat:")
        st.markdown("**Chat Bot**")
        st.caption("Free-form assistant")
        st.write(
            "A minimal chat surface for testing prompt patterns and model behavior "
            "without ceremony."
        )
        st.page_link("pages/chat_bot.py", label="Open Chat", icon=":material/arrow_forward:")

st.write("")

# ---------- Design rule ----------
with st.container(border=True):
    col_icon, col_text = st.columns([0.08, 0.92], vertical_alignment="center")
    with col_icon:
        st.markdown("### :material/hub:")
    with col_text:
        st.markdown("**One design rule**")
        st.write(
            "Lightweight ops run locally — embeddings, vector stores, chunking, scraping. "
            "LLM inference is remote via Hugging Face's Inference API. "
            "That split is what makes everything here run on a 4 GB VPS without swapping to death."
        )

# ---------- Stack ----------
st.write("")
st.caption(
    ":material/build: **Stack** · Streamlit · LangChain · Hugging Face Inference API · "
    "sentence-transformers · FAISS · Chroma · Plotly · psutil"
)