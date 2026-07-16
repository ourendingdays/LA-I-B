import streamlit as st

st.markdown("# RAG : Document Analysis :material/document_search:")
st.sidebar.markdown("##### Gentle RAG :material/document_search:")

st.sidebar.selectbox("Choose a model", ["GPT-4", "Claude", "Gemini"])
st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

st.write("Main content area")


with st.container(border=True):
    st.write("Single Document Analysis Using LLM")


with st.expander("Show details"):
    st.write("Here are the details...")
    st.image("https://static.streamlit.io/examples/dice.jpg")



tab1, tab2, tab3 = st.tabs(["Chart", "Data", "Settings"])

with tab1:
    st.line_chart({"data": [1, 5, 2, 6, 2, 1]})
with tab2:
    st.dataframe({"col1": [1, 2, 3], "col2": [4, 5, 6]})
with tab3:
    st.checkbox("Show gridlines")

with st.container(horizontal=True):
    st.text_input("Name")
    st.text_input("Email")
    st.date_input("Birthday")

with st.container(horizontal=True, horizontal_alignment="right"):
    st.button("Cancel")
    st.button("Submit")