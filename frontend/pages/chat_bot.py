# Custom Modules
from src.assistant.chat_assistant import ChatAssistant
from src.rag.config import load_config

# Standard Libraries
import streamlit as st
import time

# Initialize chat history
# We'll be adding messages to the list later on, and we don't want to overwrite the list every time the app reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------ Streamlit UI ------------
st.markdown("# Chat Bot :material/chat:")
st.sidebar.markdown("##### Gentle Chat Bot :material/chat:")


# Displaying chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("WASSSSUPPP 👋")
        st.caption("I am a gentle chat bot that can help you with your questions. Please ask me anything!")


# USER INPUT
# := operator assigns the user's input to the prompt variable and checkes if it's not None in the same line
# alternative : 
    # prompt = st.chat_input("Say something")
    # if prompt:
if prompt := st.chat_input("What is up?"):
    # Displaying user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
        st.caption(f"{time.strftime('%H:%M:%S')}")
    # Adding user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Displaying assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        st.caption(f"{time.strftime('%H:%M:%S')}")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

