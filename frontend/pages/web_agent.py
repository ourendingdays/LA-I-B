# Custom Modules
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.agents.web_search import WebSearchAgent
from src.rag.agents.research_agent import WebResearchAgent
from src.rag.config import load_config
from src.rag.visualization.charts_and_plots import create_distance_bar_chart

# Standard Libraries
import streamlit as st

if 'web_research_result' not in st.session_state:
    st.session_state.web_research_result = None
if 'web_research_processing' not in st.session_state:
    st.session_state.web_research_processing = None

def trigger_web_research():
    st.session_state.web_research_processing = "run"

@st.cache_resource(show_spinner=False)
def get_web_research_agent():
    return WebResearchAgent()


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
def get_web_search_answer(query: str, max_results: int, chosen_model: str) -> str:
    agent = WebSearchAgent()
    if chosen_model != agent.model:
        agent.change_model(chosen_model)
    content = agent.web_search(query, max_results=max_results)
    return agent.ask_model(query=content, prompt=PROMPT_TEMPLATE, model=agent.model, context=query)

def web_search(query: str, max_results: int, chosen_model: str):
    if not query:
        st.warning("Please enter a query.")
        return
    try:
        st.session_state.web_search_result = get_web_search_answer(query, max_results, chosen_model)
    except ValueError as e:
        st.error(str(e))

    
# ------------ Streamlit UI ------------
st.markdown("# Web Agents :material/robot_2:")
st.sidebar.markdown("##### Gentle Web Agents :material/robot_2:")

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


with st.expander("More Complex Web Research Agent"):
    st.caption("Searches the live web, ranks passages by semantic relevance to your question, and summarizes them.")

    web_query = st.text_input("Research question", value="What causes the long heat waves in Europe and how they originate?")
    
        
    with st.container(horizontal=True):    
        with st.container(gap="small"):
            passages_per_page = st.slider("Passages per Page", 1, 10, CONFIGURATION_WEB['web_research_agent']["passages_per_page"], step=1, help="How many text passages to keep from each fetched page.")
            top_passages = st.slider("Top Passages", 1, 15, CONFIGURATION_WEB['web_research_agent']["top_passages"], step=1, help="How many of the most relevant passages to use for the summary.")
        with st.container(gap="small"):
            summary_sentences = st.slider("Summary Sentences", 1, 10, CONFIGURATION_WEB['web_research_agent']["summary_sentences"], step=1, help="Number of sentences in the extractive summary (only used when LLM summary is off).")
            use_llm_summary = st.toggle("Use LLM summary (off = extractive, no LLM call)", value=CONFIGURATION_WEB['web_research_agent']["use_llm_summary"], help="If on, the LLM will generate a summary of the top passages. If off, an extractive summary will be generated from the top passages.")

    st.button("Research", on_click=trigger_web_research, type="primary",
              disabled=st.session_state.web_research_processing is not None)

    if st.session_state.web_research_processing == "run":
        with st.spinner("Searching, fetching, and ranking passages...", show_time=True):
            agent = get_web_research_agent()
            conf = {}
            st.session_state.web_research_result = agent.run(
                query=web_query,
                use_llm_summary=use_llm_summary,
                passages_per_page=passages_per_page,
                top_passages=top_passages,
                summary_sentences=summary_sentences,
                max_results=max_results
            )

        st.session_state.web_research_processing = None
        st.rerun()

    if st.session_state.web_research_result is not None:
        result = st.session_state.web_research_result
        st.divider()

        if result["passages"]:
            fig = create_distance_bar_chart(
                chunks=[p["passage"] for p in result["passages"]],
                distances=[1 - p["score"] for p in result["passages"]]  # convert similarity -> "distance" to reuse the existing chart
            )[0]
            st.plotly_chart(fig, width="stretch")

            with st.expander("View sources"):
                for p in result["passages"]:
                    st.markdown(f"**Score {p['score']:.3f}** — [{p['url']}]({p['url']})")
                    st.text(p["passage"])
        else:
            st.warning("No usable content found for this query.")

        st.subheader("Summary")
        st.text_area("Summary", value=result["summary"], height=150, disabled=True)


    