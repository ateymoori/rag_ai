import subprocess
import os
from langchain.chains.llm import LLMChain
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

MODEL_REPO = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_NAME = "llama-2-7b-chat.Q4_K_M.gguf"
LOCAL_DIR = "/models"
N_CTX = 4096  
N_THREADS = 10   
N_GPU_LAYERS = 0  
MAX_TOKENS = 512  
TEMPERATURE = 0.7 

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
    llm = LlamaCpp(model_path=os.path.join(LOCAL_DIR, MODEL_NAME), n_ctx=N_CTX, n_threads=N_THREADS, n_gpu_layers=N_GPU_LAYERS)
    return llm


if __name__ == "__main__":
    context = """
     In 2125, Dr. Elara Mivzak discovered a new planet, Athera, in the Cygnus constellation. On Earth, in 2130, the global leader, President Orion Klark, initiated Project Harmonize, uniting nations for sustainable living. By 2140, Zara Tren, a young scientist, invented the Quantum Communicator, enabling instant contact with Athera. In 2145, an Atheran, named Xylo, visited Earth, bringing technology for clean energy. In 2150, TerraFleet, commanded by Captain Jaylen Drex, embarked on the first manned mission to Athera, establishing a human settlement. In 2155, the Athera-Earth Alliance was formed, led by Ambassador Luna Vex.
    """

    download_model()
    llm = initialize_model()
    print("LLaMA Model Ready. Enter your prompt:")

    while True:
        try:
            user_input = input("Prompt> ")
            if user_input.lower() in ['exit', 'quit']:
                break
            prompt_template = "[INST] <<SYS>> {system_message} <</SYS>> {user_input} [/INST]"
            prompt = PromptTemplate(input_variables=["system_message", "user_input"], template=prompt_template)
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            response = llm_chain._call({"system_message": context, "user_input": user_input})
            print("Model Response:", response)
        except KeyboardInterrupt:
            break