import streamlit as st
import os
import subprocess
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_REPO = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_NAME = "llama-2-7b-chat.Q4_K_M.gguf"
LOCAL_DIR = "/models"
N_CTX = 4096  
N_THREADS = 10   
N_GPU_LAYERS = 0  
MAX_TOKENS = 256  
TEMPERATURE = 0.1 

def download_model():
    model_path = os.path.join(LOCAL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"Downloading model {MODEL_NAME} from {MODEL_REPO}...")
        command = [
            "huggingface-cli", "download", MODEL_REPO,
            MODEL_NAME, "--local-dir", LOCAL_DIR,
            "--local-dir-use-symlinks", "False"
        ]
        subprocess.run(command, check=True)
    else:
        print("Model file already exists. Skipping download.")

def initialize_model():
    llm = LlamaCpp(model_path=os.path.join(LOCAL_DIR, MODEL_NAME), n_ctx=N_CTX, n_threads=N_THREADS, n_gpu_layers=N_GPU_LAYERS,stream=True)
    return llm

def main():
    context = """In 2125, Dr. Elara Mivzak discovered a new planet, Athera, in the Cygnus constellation. On Earth, in 2130, the global leader, President Orion Klark, initiated Project Harmonize, uniting nations for sustainable living. By 2140, Zara Tren, a young scientist, invented the Quantum Communicator, enabling instant contact with Athera. In 2145, an Atheran, named Xylo, visited Earth, bringing technology for clean energy. In 2150, TerraFleet, commanded by Captain Jaylen Drex, embarked on the first manned mission to Athera, establishing a human settlement. In 2155, the Athera-Earth Alliance was formed, led by Ambassador Luna Vex."""

    download_model()
    llm = initialize_model()
    
    st.set_page_config(page_title="Llama Chatbot", layout="wide")
    st.title("Llama2 Chatbot")
    st.write("Enter your prompt below:")

    user_input = st.text_input("User Input")

    if user_input:
        prompt_template = "Context: {context}\n\nQuestion: {question} \n\nAnswer: "
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        response_placeholder = st.empty()
        response_placeholder.write("Model Response:")
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        stream_input = {"context": context, "question": user_input}
        for text in llm_chain.stream(stream_input, stop=["Q:"], callbacks=callback_manager):
            response_placeholder.write(text)
if __name__ == "__main__":
    main()
