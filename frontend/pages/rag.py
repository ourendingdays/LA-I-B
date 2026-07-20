# Custom Modules
from src.rag.hugging_face_client import HuggingFaceClient
from src.rag.simple_rag import SimpleRAG
from src.rag.utils import create_distance_bar_chart, load_config

# Standard Libraries
from pathlib import Path
import streamlit as st
import tempfile


CONFIGURATION_DATA = load_config("src/rag/configs/rag_simple.yaml")
if 'context' not in st.session_state:
    st.session_state.context = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'distances' not in st.session_state:
    st.session_state.distances = None
if 'answered_question' not in st.session_state:
    st.session_state.answered_question = None

@st.cache_data(show_spinner=False)
def get_available_hf_inference_models():
    client = HuggingFaceClient()
    return client.get_working_models(CONFIGURATION_DATA["model"].get("llm_models_to_test", []))

def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)

def answer_question(query, llm_prompt, max_tokens, chosen_model):
    """
    Generates an answer to the provided query using the RAG process and the selected LLM model.
    """
    if st.session_state.chunks is None:
        st.warning("Please retrieve text chunks first before answering a question.")
        return

    st.session_state.context = "\n\n".join(st.session_state.chunks)

    st.session_state.answered_question = get_answer(query=query, 
                                 prompt=llm_prompt, 
                                 context=st.session_state.context, 
                                 model=chosen_model,
                                 max_tokens=max_tokens)

@st.cache_data(show_spinner=True)
def get_answer(query, prompt, context, model, max_tokens):  
    """
    Generates an answer to the provided query using the RAG process and the selected LLM model.
    Guardrails: This function is cached to optimize performance. It will only re-run if the input parameters change.
    """
    rag = SimpleRAG()
    return rag.answer_question(query=query, prompt=prompt, context=context, model=model, max_tokens=max_tokens)


def display_document_info(file_path, query, llm_prompt, max_tokens, text_splitter_chunk_size, text_splitter_chunk_over, sentence_transformer_model, embeddings_top_k):
    """
    Retrieves document information based on the provided parameters : Context and distances from the RAG process.
    Wrapper: always runs, always writes to session_state — cache hit or miss.
    """
    st.session_state['chunks'], st.session_state['distances'] = get_document_info(file_path, query, llm_prompt, max_tokens,
                                           text_splitter_chunk_size, text_splitter_chunk_over,
                                           sentence_transformer_model, embeddings_top_k)
    
@st.cache_data(show_spinner=True)
def get_document_info(file_path, query, llm_prompt, max_tokens, text_splitter_chunk_size, text_splitter_chunk_over, sentence_transformer_model, embeddings_top_k):
    """
    Retrieves document information based on the provided parameters : Context and distances from the RAG process.
    Guardrailes: This function is cached to optimize performance. It will only re-run if the input parameters change.
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
    chunks, distances = rag.preprocess_document(file_path=file_path, query=query, configuration_data=configuration_data)
    return chunks, distances

# ------------ Streamlit UI ------------
st.markdown("# RAG : Document Analysis :material/document_search:")
st.sidebar.markdown("##### Gentle RAG :material/document_search:")

with st.container(border=True, gap="small"):
    st.write("Available Models for Hugging Face Inference API right now.")

    st.link_button(label="See more", url="https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation&sort=trending", type="tertiary")

    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        with st.spinner("Looking for available models...", show_time=True):
            chosen_model_predefined = st.selectbox("Available models at the moment: ", get_available_hf_inference_models())
    with col2:
        chosen_model = st.text_input("Enter your own (must be an Instruct with chat_completion):", value=chosen_model_predefined)

SIMPLE_RAG_TAB, GRAPH_RAG = st.tabs(["Simplest RAG", "LLM Analysis"])

with SIMPLE_RAG_TAB:
    st.caption("Single Document Analysis Using LLM. It retrieves relevant chunks from a document and generates an answer using a language model.")
    
    # ------------ File Upload and Query Input ------------
    uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf"])
    if uploaded_file is not None:
        st.write("File uploaded:", uploaded_file.name)
        file_path = save_uploaded_file(uploaded_file)
    else:
        st.write("No file uploaded yet. Please upload a document to proceed.")
        file_path = Path("data/raw/txt/rag_notebook.txt")  # Default file path if no file is uploaded

    llm_prompt = st.text_area(label='LLM Prompt', value=CONFIGURATION_DATA["model"].get("llm_prompt"))
    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        query = st.text_input("Query", value="What is the distance from Earth to the Moon?", help="The question you want to ask based on the document's content.")
    with col2:
        max_tokens = st.slider("Max Tokens", 150, 2000, 200, step=50, help="Maximum number of tokens for the generated answer.")

    # ------------ Configuration Parameters ------------
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
        st.button("Retrieve Text Chunks", 
                  on_click=display_document_info, 
                  args=(file_path, query, llm_prompt, max_tokens, text_splitter_chunk_size, text_splitter_chunk_over, sentence_transformer_model, embeddings_top_k), 
                  type="primary")
        st.button("Answer Question", on_click=answer_question, 
            args=(query, llm_prompt, max_tokens, chosen_model), type="primary")

    # ----------- Display Retrieved Chunks with Distances And Final Answer ------------
    if st.session_state.chunks is not None and st.session_state.distances is not None:
        fig, order = create_distance_bar_chart(st.session_state.chunks, st.session_state.distances)
        st.plotly_chart(fig, width="stretch")

        with st.expander("View full chunk text"):
            for i in order:
                st.markdown(f"**Chunk {i + 1}** — distance: `{st.session_state.distances[i]:.4f}`")
                st.text(st.session_state.chunks[i])

    if st.session_state.answered_question is not None:
        st.divider()
        st.text_area("Answer", value=st.session_state.answered_question, height=120, disabled=True)

with GRAPH_RAG:
    st.dataframe({"col1": [1, 2, 3], "col2": [4, 5, 6]})