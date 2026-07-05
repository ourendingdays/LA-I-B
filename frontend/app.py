import streamlit as st


st.set_page_config(
    page_title="RA(I)G",
    page_icon="📜",
)

# Define the pages
page_main = st.Page("pages/main_page.py", title="Home", icon="📚")
page_doc_analysis = st.Page("pages/document_analysis.py", title="Document Analysis", icon="📄")

# Set up navigation
pg = st.navigation([page_main, page_doc_analysis])

# Run the selected page
pg.run()