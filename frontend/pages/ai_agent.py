# Custom Modules
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.agents.web_search import WebSearchAgent
from src.rag.config import load_config

# Standard Libraries
import streamlit as st

PROMPT_TEMPLATE = """You are a helpful AI assistant. Answer the user's question
                based *only* on the following search results. If the search results
                are empty or do not contain the answer, say 'I could not find
                any information on that.'"""

CONFIGURATION_RAG = load_config("src/rag/configs/rag_simple.yaml")
CONFIGURATION_WEB = load_config("src/rag/configs/web_agent.yaml")

if 'web_search_result' not in st.session_state:
    st.session_state.web_search_result = None

@st.cache_data(show_spinner=False)
def get_available_hf_inference_models():
    client = HuggingFaceClient()
    return client.get_working_models(CONFIGURATION_RAG["model"].get("instruct_completion_models", []))

@st.cache_data(show_spinner=True)
def web_search(query: str, max_results: int, chosen_model: str):
    """
    Performs a web search using DuckDuckGo and answers the query using a Hugging Face Inference API model.

    Args:
        query (str): The user's query.
        max_results (int): Maximum number of web search results to retrieve.
        chosen_model (str): The Hugging Face model to use for answering the query.
    """
    agent = WebSearchAgent()

    if not query:
        st.warning("Please enter a query.")
        return
    
    # Change the model if it's different from the current one
    if chosen_model != agent.model:
        try:
            agent.change_model(chosen_model)
        except ValueError as e:
            st.error(str(e))
            return

    content = agent.web_search(query)
    st.session_state.web_search_result = agent.ask_model(query = content, 
                           prompt = PROMPT_TEMPLATE,  
                           model = agent.model, 
                           context = query)

    
# ------------ Streamlit UI ------------
st.markdown("# AI Agents :material/robot_2:")
st.sidebar.markdown("##### Gentle AI Agents :material/robot_2:")

with st.container(border=True, gap="small"):

    st.write("Available Models for Hugging Face Inference API right now.", "[Find more here](https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation&sort=trending)")
    
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        with st.spinner("Looking for available models...", show_time=True):
            chosen_model_predefined = st.selectbox("Available models at the moment: ", get_available_hf_inference_models())
    with col2:
        chosen_model = st.text_input("Enter your own (must be an Instruct with chat_completion):", value=chosen_model_predefined)

st.caption("Web Search Agent. It uses a DuckDuckGo & Hugging Face Inference API model to answer questions based on web search results. ")

col1, col2 = st.columns([0.75, 0.25])
with col1:
    query = st.text_input("Query", value="Who was in Apollo 11 crew?", help="The question you want to search the internet for.")
with col2:
    max_results = st.slider("Max Results", 1, 30, CONFIGURATION_WEB["web_search"]["max_results"], step=1, help="Maximum number of web search resultsto retrieve.")

with st.container(horizontal=True, horizontal_alignment="right"):
        st.button("Search", on_click=web_search, args=(query, max_results, chosen_model), type="primary")

if st.session_state.web_search_result is not None:
    st.markdown("### Web Search Result:")
    st.text_area("Result", value=st.session_state.web_search_result, height=200)