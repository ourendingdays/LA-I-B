# Custom Modules
from src.rag.hugging_face_client import HuggingFaceClient
from src.rag.simple_rag import SimpleRAG
from src.rag.utils import load_config

# Standard Libraries
from pathlib import Path
import streamlit as st
import tempfile


CONFIGURATION_DATA = load_config("src/rag/configs/rag_simple.yaml")
if 'context' not in st.session_state:
    st.session_state.context = None

if 'distances' not in st.session_state:
    st.session_state.distances = None

@st.cache_data(show_spinner=False)
def get_available_hf_inference_models():
    client = HuggingFaceClient()
    return client.get_working_models(CONFIGURATION_DATA.get("llm_models_to_test", []))

def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)
    
@st.cache_data(show_spinner=True)
def display_document_info(file_path, query, llm_prompt, max_tokens, text_splitter_chunk_size, text_splitter_chunk_over, sentence_transformer_model, embeddings_top_k):
    """
    Retrieves document information based on the provided parameters : Context and distances from the RAG process.
    """
    configuration_data = {
        "llm_prompt": llm_prompt,
        "ts_chunk_size": text_splitter_chunk_size,
        "ts_chunk_overlap": text_splitter_chunk_over,
        "sentence_transformer_model": sentence_transformer_model,
        "embeddings_top_k": embeddings_top_k,
        "llm_max_tokens": max_tokens
    }

    rag = SimpleRAG()
    st.session_state['context'], st.session_state['distances'] = rag.run(file_path=file_path, query=query, configuration_data=configuration_data)


st.markdown("# RAG : Document Analysis :material/document_search:")
st.sidebar.markdown("##### Gentle RAG :material/document_search:")

with st.container(border=True):
    st.write("Available Models for Hugging Face Inference API right now.")
    
    # st.caption("Go to this URL to see every model available through inference providers, filtered to text generation and sorted by trending")
    custom_text = "<p style='font-size: 12px; color: gray;'>Go to this URL to see every model available through inference providers, filtered to text generation and sorted by trending</p>"
    st.markdown(custom_text, unsafe_allow_html=True)

    st.link_button("Hugging Face Models", "https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation&sort=trending")

    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        with st.spinner("Looking for available models...", show_time=True):
            chosen_model_predefined = st.selectbox("Available models at the moment: ", get_available_hf_inference_models())
    with col2:
        chosen_model = st.text_input("Enter your own (must be an Instruct with chat_completion):", value=chosen_model_predefined)

SIMPLE_RAG_TAB, tab2 = st.tabs(["Simplest RAG", "Data"])

with SIMPLE_RAG_TAB:
    st.caption("Single Document Analysis Using LLM. It retrieves relevant chunks from a document and generates an answer using a language model.")
    uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf"])
    if uploaded_file is not None:
        st.write("File uploaded:", uploaded_file.name)
        file_path = save_uploaded_file(uploaded_file)
    else:
        st.write("No file uploaded yet. Please upload a document to proceed.")
        file_path = Path("data/raw/txt/rag_notebook.txt")  # Default file path if no file is uploaded

    st.divider()
    llm_prompt = st.text_area(label='LLM Prompt', value=CONFIGURATION_DATA.get("llm_prompt"))
    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        query = st.text_input("Query", value="What is the distance from Earth to the Moon?", help="The question you want to ask based on the document's content.")
    with col2:
        max_tokens = st.slider("Max Tokens", 150, 2000, 200, step=50, help="Maximum number of tokens for the generated answer.")

    with st.expander("Configuration Parameters"):
        with st.container(horizontal=True):    
            with st.container(gap="small"):
                text_splitter_chunk_size = st.slider("Chunk Size", 50, 500, 150, step=50, help="Size of each text chunk.")
                text_splitter_chunk_over = st.slider("Chunk Overlap", 0, 100, 20, step=5, help="Overlap between text chunks.")
            with st.container(gap="small"):
                sentence_transformer_model = st.text_input("Sentence Transformer Model", 
                                                           value=CONFIGURATION_DATA.get("sentence_transformer_model", "all-MiniLM-L6-v2"),
                                                           help="Model used for generating embeddings for text chunks.")
                embeddings_top_k = st.slider("Top K", 1, 10, 3, step=1, help="Number of top relevant chunks to retrieve.")

        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button("Retrieve Text Chunks", on_click=display_document_info, 
                    args=(file_path, query, llm_prompt, max_tokens, text_splitter_chunk_size, text_splitter_chunk_over, sentence_transformer_model, embeddings_top_k), type="primary")

    if st.session_state.context is not None and st.session_state.distances is not None:
        st.divider()
        st.subheader("Retrieved Context")
        st.text_area("Context", value=st.session_state.context, height=300, disabled=True)
        st.subheader("Distances of Retrieved Chunks")
        st.text_area("Distances", value=str(st.session_state.distances), height=100, disabled=True)



with tab2:
    st.dataframe({"col1": [1, 2, 3], "col2": [4, 5, 6]})