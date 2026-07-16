import streamlit as st
import time
from src.utils.hugging_face_client import HuggingFaceClient

# How to pick a good model:
# You want models that are "Instruct" tuned (they follow instructions) and support chat_completion. Look for these patterns in the name: Instruct, it (Google's naming), Chat. Avoid base models (no instruction tuning — they just autocomplete text, not answer questions).
MODELS_TO_TEST = [
    # Qwen family
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    
    # Meta Llama family
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    
    # Google Gemma family
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    
    # Mistral family
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    
    # Microsoft Phi family
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-4-mini-instruct",
    
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    
    # Others
    "HuggingFaceH4/zephyr-7b-beta",
    "NousResearch/Hermes-3-Llama-3.1-8B",
]

@st.cache_resource
def get_available_hf_inference_models():
    client = HuggingFaceClient()
    return client.get_working_models(MODELS_TO_TEST)


st.markdown("# RAG : Document Analysis :material/document_search:")
st.sidebar.markdown("##### Gentle RAG :material/document_search:")

st.sidebar.selectbox("Choose a model", ["GPT-4", "Claude", "Gemini"])
st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)


with st.container(border=True):
    st.write("Single Document Analysis Using LLM")


with st.container(border=True):
    st.write("Available Models for Hugging Face Inference API right now.")
    
    # st.caption("Go to this URL to see every model available through inference providers, filtered to text generation and sorted by trending")
    custom_text = "<p style='font-size: 12px; color: gray;'>Go to this URL to see every model available through inference providers, filtered to text generation and sorted by trending</p>"
    st.markdown(custom_text, unsafe_allow_html=True)

    st.link_button("Hugging Face Models", "https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation&sort=trending")

    col1, col2 = st.columns(2)
    with col1:
        chosen_model_predefined = st.selectbox("These are the available models at the moment: ", get_available_hf_inference_models())
    with col2:
        chosen_model = st.text_input("Or enter your own (must be an Instruct model that supports chat_completion):", value=chosen_model_predefined)


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


tab1, tab2 = st.tabs(["Chart", "Data"], on_change="rerun")

if tab1.open:
    with st.spinner("Loading Tab 1..."):
        time.sleep(2)
    with tab1:
        st.line_chart({"data": [1, 5, 2, 6]})

if tab2.open:
    with st.spinner("Loading Tab 2..."):
        time.sleep(2)
    with tab2:
        st.dataframe({"col1": [1, 2, 3]})

def on_tab_change():
    st.toast(f"Tab changed to {st.session_state.tabs}!")

tab1, tab2 = st.tabs(["Input", "Output"], on_change=on_tab_change, key="tabs")