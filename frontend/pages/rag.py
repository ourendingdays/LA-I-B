# Custom Modules
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.simple_doc_analyzer import SimpleDocumentAnalyzer
from src.rag.simple_graphrag import GraphRAG
from src.rag.simple_rag import SimpleRAG
from src.rag.config import load_config
from src.rag.documents.loaders import load_document
from src.rag.utils import create_distance_bar_chart, create_embedding_scatter, create_graph_visualization

# Standard Libraries
from pathlib import Path
import streamlit as st
import tempfile


CONFIGURATION_DATA = load_config("src/rag/configs/rag_simple.yaml")

# --- RAG
if 'context' not in st.session_state:
    st.session_state.context = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'distances' not in st.session_state:
    st.session_state.distances = None
if 'answered_question' not in st.session_state:
    st.session_state.answered_question = None
if 'embedding_viz' not in st.session_state:
    st.session_state.embedding_viz = None

def get_summary(file_path: Path, model: str) -> str:
    doc_analyzer = SimpleDocumentAnalyzer()
    query = "What is the main topic of the document?"

    knowledge_text  = load_document(file_path = file_path)
    
    summary = doc_analyzer.summarize_document(input_text=knowledge_text, 
                                    model=CONFIGURATION_DATA.get("model", {}).get("summarization_models", ["facebook/bart-large-cnn"])[0])
    return summary

@st.cache_data(show_spinner=False)
def get_available_hf_inference_models():
    client = HuggingFaceClient()
    return client.get_working_models(CONFIGURATION_DATA["model"].get("instruct_completion_models", []))

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
    return rag.ask_model(query=query, prompt=prompt, context=context, model=model, max_tokens=max_tokens)

def display_document_info(file_path: Path, query: str, llm_prompt: str, max_tokens: int, text_splitter_chunk_size: int, text_splitter_chunk_over: int, sentence_transformer_model: str, embeddings_top_k: int):
    """
    Retrieves document information based on the provided parameters : Context and distances from the RAG process.
    Wrapper: always runs, always writes to session_state — cache hit or miss.
    """
    (st.session_state['chunks'], st.session_state['distances'], all_chunks, chunk_embeddings, query_embedding, retrieved_indices) = get_document_info(
                                        file_path, query, llm_prompt, max_tokens,
                                           text_splitter_chunk_size, text_splitter_chunk_over,
                                           sentence_transformer_model, embeddings_top_k)
    
    st.session_state['embedding_viz'] = {
        "all_chunks": all_chunks,
        "chunk_embeddings": chunk_embeddings,
        "query_embedding": query_embedding,
        "retrieved_indices": retrieved_indices,
    }

@st.cache_data(show_spinner=True)
def get_document_info(file_path, query, llm_prompt, max_tokens, text_splitter_chunk_size, text_splitter_chunk_over, sentence_transformer_model, embeddings_top_k):
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
    all_chunks, chunk_embeddings, query_embedding, retrieved_indices = rag.get_embedding_visualization_data()
    return chunks, distances, all_chunks, chunk_embeddings, query_embedding, retrieved_indices


# --- GraphRAG
if 'kg' not in st.session_state:
    st.session_state.kg = None
if 'triples' not in st.session_state:
    st.session_state.triples = None
if 'graph_result' not in st.session_state:
    st.session_state.graph_result = None
if 'graph_processing' not in st.session_state:
    st.session_state.graph_processing = None

def trigger_build_graph():
    st.session_state.graph_processing = "build"

def trigger_ask_graph():
    st.session_state.graph_processing = "ask"

@st.cache_resource(show_spinner=False)
def get_graph_rag_client():
    return GraphRAG()

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

SIMPLE_RAG_TAB, GRAPH_RAG = st.tabs(["Simplest RAG", "Graph RAG"])

with SIMPLE_RAG_TAB:
    st.caption("Single Document Analysis Using LLM. It retrieves relevant chunks from a document and generates an answer using a language model.")
    
    # ------------ File Upload and Query Input ------------
    uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf"])
    if uploaded_file is not None:
        st.write("File uploaded:", uploaded_file.name)
        file_path = save_uploaded_file(uploaded_file)

        summary = get_summary(file_path=file_path, model=chosen_model)
        st.text_area("Summary", value=summary, height=80, disabled=True)
       
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

    if st.session_state.embedding_viz is not None:
        st.subheader("Embedding Space")
        st.caption("2D projection (PCA) of all document chunks. Blue = retrieved for this query, red star = your query.")
        viz = st.session_state.embedding_viz
        scatter_fig = create_embedding_scatter(
            chunk_embeddings=viz["chunk_embeddings"],
            query_embedding=viz["query_embedding"],
            chunks=viz["all_chunks"],
            retrieved_indices=viz["retrieved_indices"],
        )
        st.plotly_chart(scatter_fig, width="stretch")

    if st.session_state.answered_question is not None:
        st.divider()
        st.text_area("Answer", value=st.session_state.answered_question, height=120, disabled=True)

with GRAPH_RAG:
    st.caption("Graph RAG is is a network of entities (nodes) and relationships (edges) - it views a text as a network of connected facts.")
    st.text("Extracts entities and relationships into a knowledge graph, then answers questions via multi-hop graph traversal — good for connecting facts that vector search alone can't link.")

    graph_text = st.text_area(
        "Text to build the knowledge graph from",
        value="The Moon orbits Earth. The Moon has an atmosphere called the Exosphere. "
              "Apollo 11 landed on the Moon. The Moon has a crater named the South Pole-Aitken Basin. "
              "Earth's Moon is classified as a natural satellite.",
        height=120
    )

    st.button("Build Knowledge Graph", on_click=trigger_build_graph, type="primary", disabled=st.session_state.graph_processing is not None)

    if st.session_state.graph_processing == "build":
        with st.spinner("Extracting entities and building graph...", show_time=True):
            graph_rag = get_graph_rag_client()
            kg, triples = graph_rag.build_knowledge_graph(text=graph_text, model=chosen_model)
            st.session_state.kg = kg
            st.session_state.triples = triples
            st.session_state.graph_result = None  # clear any stale answer from a previous graph
        st.session_state.graph_processing = None
        st.rerun()

    if st.session_state.kg is not None:
        st.divider()
        st.subheader("Knowledge Graph")

        with st.expander("Extracted triples"):
            st.dataframe(
                [{"Head": t.get("head"), "Relation": t.get("relation"), "Tail": t.get("tail")} for t in st.session_state.triples],
                width="stretch"
            )

        # Highlight the last traversal path, if there is one, otherwise show the plain graph
        highlight_nodes = st.session_state.graph_result["visited_nodes"] if st.session_state.graph_result else None
        highlight_edges = st.session_state.graph_result["traversed_edges"] if st.session_state.graph_result else None

        graph_html = create_graph_visualization(st.session_state.kg, highlight_nodes, highlight_edges)
        st.components.v1.html(graph_html, height=520)

        st.divider()
        st.subheader("Ask the Graph")

        entities = list(st.session_state.kg.nodes())
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            graph_question = st.text_input("Question", value="On which natural satellite did Apollo land?")
        with col2:
            manual_entity = st.selectbox("Starting entity (optional — auto-detected if left blank)", ["(auto-detect)"] + entities)

        graph_max_depth = st.slider("Traversal depth (multi-hop distance)", 1, 5, 3)

        st.button("Ask Graph", on_click=trigger_ask_graph, type="primary",
                  disabled=st.session_state.graph_processing is not None)

        if st.session_state.graph_processing == "ask":
            with st.spinner("Traversing graph and generating answer...", show_time=True):
                graph_rag = get_graph_rag_client()
                entity_arg = None if manual_entity == "(auto-detect)" else manual_entity
                st.session_state.graph_result = graph_rag.graph_rag_answer(
                    question=graph_question,
                    model=chosen_model,
                    entity=entity_arg,
                    max_depth=graph_max_depth
                )
            st.session_state.graph_processing = None
            st.rerun()

        if st.session_state.graph_result is not None:
            st.divider()
            result = st.session_state.graph_result
            st.markdown(f"**Starting entity:** `{result['entity']}`")
            with st.expander("Retrieved graph context"):
                st.text(result["graph_context"] or "(no context found)")
            st.text_area("Answer", value=result["answer"], height=100, disabled=True)
    else:
        st.info("Build a knowledge graph above to get started.")