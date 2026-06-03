import streamlit as st
from dotenv import load_dotenv
from src.pipeline import MusicRAGPipeline

# 1. Load environment variables (.env) and cache the RAG pipeline
load_dotenv()

@st.cache_resource
def init_pipeline():
    # This prevents Streamlit from re-loading the heavy FAISS index 
    # every time you click a button or type text
    return MusicRAGPipeline()

try:
    pipeline = init_pipeline()
except Exception as e:
    st.error(f"Failed to initialize RAG pipeline. Did you run ingestion first? Error: {e}")
    st.stop()

# 2. App Window Configuration
st.set_page_config(page_title="AI Musicologist", page_icon="🎵", layout="centered")

st.title("🎵 AI Musicologist & Crate-Digger")
st.caption("Ask questions about your album collection, production styles, or get tailored mood recommendations.")

# 3. Initialize Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "What kind of vibe or production style are you looking for today?"}
    ]

# 4. Display Past Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle New User Input
if user_query := st.chat_input("Search your music vault..."):
    # Display user message instantly
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate response using your pipeline with a nice loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Digging through the crates..."):
            try:
                response = pipeline.query(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")