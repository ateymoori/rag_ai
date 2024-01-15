import streamlit as st
import os
import subprocess
from llama_cpp import Llama

MODEL_REPO = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_NAME = "llama-2-7b-chat.Q4_K_M.gguf"
LOCAL_DIR = "models"
N_CTX = 4096
MAX_TOKENS = 512
TEMPERATURE = 0.7

st.set_page_config(page_title="Llama Chatbot", layout="wide")

@st.cache_resource
def download_model():
    model_path = os.path.join(LOCAL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        subprocess.run(
            ["huggingface-cli", "download", MODEL_REPO, "-n", MODEL_NAME, "--cache-dir", LOCAL_DIR],
            check=True
        )

@st.cache_resource
def load_model():
    download_model()
    model_path = os.path.join(LOCAL_DIR, MODEL_NAME)
    return Llama(model_path=model_path, chat_format="llama-2")

llm = load_model()

def app():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    with st.sidebar:
        st.header("Settings")
        temperature = st.slider('Temperature', 0.01, 5.0, value=TEMPERATURE, step=0.01)
        top_p = st.slider('Top P', 0.01, 1.0, value=0.9, step=0.01)
        max_length = st.slider('Max Length', 32, max_value=N_CTX, value=MAX_TOKENS, step=8)


    # Display chat history
    for index, (role, message) in enumerate(st.session_state['chat_history']):
        st.text_area(f"{role.title()} says:", value=message, height=75, disabled=True, key=f"chat_message_{index}")

    # Text input at the bottom
    user_message = st.text_input('Enter your message:', key='user_message_input', on_change=generate_response_and_update_history, args=(llm, temperature, top_p, max_length))


def generate_response_and_update_history(llm, temperature, top_p, max_length):
    user_message = st.session_state.user_message_input
    if user_message:
        st.session_state['chat_history'].append(('user', user_message))
        response = generate_llama2_response(llm, temperature, top_p, max_length)
        st.session_state['chat_history'].append(('bot', response))
        st.session_state.user_message_input = ''  # Clear the input field
        print_chat_history()

def print_chat_history():
    print("Chat History:")
    for role, message in st.session_state['chat_history']:
        print(f"{role.title()}: {message}")

def generate_llama2_response(llm, temperature, top_p, max_length):
    chat_history_formatted = [
        {"role": role, "content": message}
        for role, message in st.session_state['chat_history']
    ]
    response = llm.create_chat_completion(
        messages=chat_history_formatted,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_length
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    app()
