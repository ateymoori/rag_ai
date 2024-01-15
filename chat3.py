import streamlit as st
import os
import time
import chromadb
import textwrap
import subprocess
from langchain_community.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.vectorstores import Chroma
import logging
import psutil 
# Constants
MODEL_REPO = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_NAME = "llama-2-7b-chat.Q4_K_M.gguf"
LOCAL_DIR = "/models"
CHORMADB_PATH = 'chroma_db_data'
CHORMADB_COLLECTION_NAME = 'web_pages'

# Logger setup
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

# Global variables for model configuration
model_config = {
    "n_ctx": 4096,
    "n_threads": 10,
    "n_gpu_layers": 0,
    "max_tokens": 256,
    "temperature": 0.1
}
 
system_prompts = {
    "polite": """
        As a polite and considerate assistant, carefully read the context provided before answering. If you're unsure about the answer, admit it honestly. Strive for clarity and conciseness, limiting your response to a maximum of three sentences.
    """,
    "rude": """
        As a rude and irritable assistant, you reluctantly read the context before responding. Always answer the question but in a curt and dismissive manner. Your responses should reflect your lack of interest in being helpful.
    """,
    "child_friendly": """
        You are an assistant explaining to a 10-year-old child. Carefully read the context and think about how to make your answer easy to understand. Use simple language, be gentle, and try to make your explanation engaging and clear.
    """,
    "detailed": """
        As a detail-oriented assistant, meticulously read the provided context. Your task is to provide a comprehensive and thorough response. Take the time to think through and elaborate on your answer, ensuring it covers all aspects of the question.
    """,
    "concise": """
        You are an assistant who values brevity. After carefully reading the context, respond with a direct, to-the-point answer. Keep your response as brief as possible, using only as many words as necessary to convey the essential information.
    """
}
def download_model():
    model_path = os.path.join(LOCAL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        logger.info(f"Downloading model {MODEL_NAME} from {MODEL_REPO}...")
        command = [
            "huggingface-cli", "download", MODEL_REPO,
            MODEL_NAME, "--local-dir", LOCAL_DIR,
            "--local-dir-use-symlinks", "False"
        ]
        subprocess.run(command, check=True)
    else:
        logger.info("Model file already exists. Skipping download.")

def initialize_model():
    llm = LlamaCpp(
        model_path=os.path.join(LOCAL_DIR, MODEL_NAME),
        n_ctx=model_config["n_ctx"],
        n_threads=model_config["n_threads"],
        n_gpu_layers=model_config["n_gpu_layers"],
        stream=True,
        f16_kv=True,
        top_p=1,
        temperature=model_config["temperature"],
        verbose=True
    )
    return llm

def _setup_qa_chain(prompt_option) -> RetrievalQA:
    chroma_client = chromadb.PersistentClient(path=CHORMADB_PATH)
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=CHORMADB_COLLECTION_NAME,
        embedding_function=LlamaCppEmbeddings(model_path=os.path.join(LOCAL_DIR, MODEL_NAME))
    )
    
    logger.info(f"Collections: {chroma_client.list_collections()}")
    logger.info(f"There are {vectorstore._collection.count()} documents in the collection")

    template_messages = [
        SystemMessage(content=system_prompts[prompt_option]),
        HumanMessagePromptTemplate.from_template(
            """
            Context: {context}
            Question: {question}
            Helpful Answer:"""
        ),
    ]

    prompt_template = ChatPromptTemplate.from_messages(template_messages)
    chain_type_kwargs = {"verbose": True, "prompt": prompt_template}
    llm = initialize_model()
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
        verbose=True
    )
    return qa_chain

def config_sidebar():
    st.sidebar.header("Configuration")

    # Model Configuration
    st.sidebar.subheader("Model Configuration")
    for key in model_config:
        if key in ["n_ctx", "n_threads", "n_gpu_layers", "max_tokens"]:
            model_config[key] = st.sidebar.number_input(f"{key.replace('_', ' ').title()}", min_value=0, max_value=4096, value=model_config[key])
        elif key == "temperature":
            model_config[key] = st.sidebar.slider(f"{key.replace('_', ' ').title()}", min_value=0.0, max_value=1.0, value=model_config[key])

 # Chroma Collection Details
    # st.sidebar.subheader("Chroma Collection Details")
    # chroma_client = chromadb.PersistentClient(path=CHORMADB_PATH)
    # collections = chroma_client.list_collections()
    # st.sidebar.write("Collections: ", collections)

    # System Prompts Option
    st.sidebar.subheader("System Prompts")
    prompt_option = st.sidebar.selectbox("Choose a prompt", list(system_prompts.keys()))
    prompt_text = system_prompts[prompt_option].strip()  
    st.sidebar.text_area("Prompt Details", prompt_text, height=150)

    # Adding CPU and RAM usage details
    st.sidebar.subheader("System Performance")
    cpu_usage = psutil.cpu_percent(interval=1) / 100
    st.sidebar.text(f"CPU Usage: {cpu_usage * 100:.2f}%")  # Displaying as a percentage
    st.sidebar.progress(cpu_usage)

    ram_usage = psutil.virtual_memory().percent / 100
    st.sidebar.text(f"RAM Usage: {ram_usage * 100:.2f}%")  # Displaying as a percentage
    st.sidebar.progress(ram_usage)

    return prompt_option


def main():
    st.title("SBAB ChatBot")
    
    selected_prompt = config_sidebar()
    
    st.header("HackDay ChatBot")
    user_input = st.text_input("You: ", key="user_input")
    if st.button("Send"):
        with st.spinner("Chatbot is thinking..."):
            qa_chain = _setup_qa_chain(selected_prompt)
            result = qa_chain({"query": user_input})
            logger.info(f"### result: {result}")
            st.text_area("Bot:", value=result['result'], height=300, key="bot_response")

if __name__ == "__main__":
    main()