import streamlit as st
from model.load_model import load_model_and_tokenizer
from app.chatbot_logic import generate_response

st.set_page_config(page_title="Philosophy Bot", page_icon="🧠", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Chatbot"])

# Load model and tokenizer once
@st.cache_resource
def load_model_once():
    return load_model_and_tokenizer()

model, tokenizer = load_model_once()

# Session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

if page == "Home":
    st.title("🧠 Philosophy Bot")
    st.write("Welcome to Philosophy Bot!")
    st.write("""
    This chatbot has been fine-tuned on philosophical articles using LoRA on the LLaMA 2 7B model.
    You can have deep, thoughtful conversations here. 🌟
    """)
    st.markdown("---")
    st.info("Click on **Chatbot** in the sidebar to start chatting! ➡️")

elif page == "Chatbot":
    st.title("💬 Chatbot")
    user_input = st.text_input("You:", "")

    if user_input:
        response, updated_history = generate_response(model, tokenizer, st.session_state.history, user_input)
        st.session_state.history = updated_history
        st.success(f"Assistant: {response}")

    # Display conversation history
    if st.session_state.history:
        st.markdown("---")
        for message in st.session_state.history:
            st.write(message)
