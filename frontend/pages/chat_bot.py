# Custom Modules
from src.assistant.chat_assistant import ChatAssistant
from src.rag.config import load_config

# Standard Libraries
from pathlib import Path
import random
import streamlit as st
import tempfile
import time

GREETING_EMOJIS = [
    "🌇", "🌅", "🌆", "🎡", "🚀", "🚂", "🍮", "🎱",
    "🎉", "✨", "🎭", "⭐", "🚀", "🎊", "🔥", "🎈", "🍿", "🧃", 
    "🤖", "🌑", "🌗", "🌘 "
]

ANIMAL_EMOJIS = [
    "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯", "🦁", "🐮",
    "🐷", "🐸", "🐵", "🐔", "🐧", "🐦", "🐤", "🦆", "🦉", "🐺", "🐴", "🦄",
    "🐝", "🦋", "🐌", "🐞", "🐢", "🐙", "🦑", "🐠", "🐬", "🐳", "🐋", "🦥",
    "🦦", "🦔", "🦭", "🐿️", "🦫", "🦩", "🦢", "🦜", "🕊️",
]

CHAT_GREETINGS = [
    "WASSSSUPPP",
    "YOOOO",
    "HELLO THERE",
    "HEYYY",
    "AYYY",
    "HOWDY",
    "AHOY",
    "OI OI",
    "SUP",
    "WHADDUP",
    "HIYA",
]


# Initialize chat history
# We'll be adding messages to the list later on, and we don't want to overwrite the list every time the app reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Picked once per browser session so the emojis don't reshuffle on every rerun.
# Fresh session (tab refresh, new tab) → new picks.
if "greeting_emoji" not in st.session_state:
    st.session_state.greeting_emoji = random.choice(GREETING_EMOJIS)
if "greeting_phrase" not in st.session_state:
    st.session_state.greeting_phrase = random.choice(CHAT_GREETINGS)
if "user_animal" not in st.session_state:
    st.session_state.user_animal, st.session_state.assistant_animal = random.sample(ANIMAL_EMOJIS, 2)

def get_time_greetting():
    """Returns a greeting based on the current time of day."""
    current_hour = time.localtime().tm_hour
    if 5 <= current_hour < 10:
        return "Good morning! ☀️"
    elif 10 <= current_hour < 18:
        return "Good day! 🌤️"
    elif 18 <= current_hour < 22:
        return "Good evening! 🌙"
    else:
        return "Hello! 🌌"


def _animal_for(role: str) -> str:
    return st.session_state.user_animal if role == "user" else st.session_state.assistant_animal

def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)

# Cached so the ChatAssistant (and its "Loading weights") is built ONCE per session,
# not on every rerun.
@st.cache_resource(show_spinner="Warming up the assistant...")
def get_assistant():
    return ChatAssistant()

assistant = get_assistant()

def answer(uploaded_file, prompt):
    """
    Answers the user's query using the ChatAssistant.
    If a new file is attached, it re-indexes; otherwise it reuses the last-indexed doc.
    """
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        docs = assistant.load_file(file_path)
        assistant.process_source_data(docs)
    answer = assistant.answer_query(prompt)
    return answer

# ------------ Streamlit UI ------------
st.markdown("# Chat Bot :material/chat:")
st.sidebar.markdown("##### Gentle Chat Bot :material/chat:")
st.caption(f"{get_time_greetting()} This is a simple RAG-based chat bot. Ask away!")


# Displaying chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "timestamp" in message:
            st.caption(f"{_animal_for(message['role'])} · {message['timestamp']}")


if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write(f"{st.session_state.greeting_phrase} {st.session_state.greeting_emoji}")
        st.caption("I am a gentle chat bot that can help you with your questions. Please ask me anything!")


# USER INPUT
# := operator assigns the user's input to the prompt variable and checkes if it's not None in the same line
# Bundled input: text + optional file attachment in the SAME chat input widget (paperclip icon).
# alternative : 
    # prompt = st.chat_input("Say something")
    # if prompt:
if user_input := st.chat_input("What is up?", accept_file=True, file_type=["txt", "pdf"]):
    prompt = user_input.text
    uploaded_file = user_input.files[0] if user_input.files else None

    now = time.strftime("%H:%M:%S")
    # Displaying user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
        if uploaded_file is not None:
            st.caption(f"📎 {uploaded_file.name}")
        st.caption(f"{st.session_state.user_animal} · {now}")
    # Adding user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": now})

    if uploaded_file is None and not st.session_state.get("doc_indexed"):
        response = "Please attach a document (📎 in the input) so I can answer questions about it."
    else:
        if uploaded_file is not None:
            st.session_state.doc_indexed = True
        with st.spinner("Reading the document and thinking..."):
            resp = answer(uploaded_file, prompt)
        response = f"{resp}"

    # Displaying assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        st.caption(f"{st.session_state.assistant_animal} · {now}")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": now})