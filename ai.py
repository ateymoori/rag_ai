import subprocess
import os
from llama_cpp import Llama

# Configuration Variables
MODEL_REPO = "TheBloke/LLaMA-Pro-8B-Instruct-GGUF"
MODEL_NAME = "llama-pro-8b-instruct.Q4_K_M.gguf"
LOCAL_DIR = "/models"
N_CTX = 4096  # Max sequence length
N_THREADS = 10  # Number of CPU threads
N_GPU_LAYERS = 0  # No GPU layers
MAX_TOKENS = 512  # Max tokens for generation
TEMPERATURE = 0.7  # Temperature setting

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
    llm = Llama(model_path=os.path.join(LOCAL_DIR, MODEL_NAME), n_ctx=N_CTX, n_threads=N_THREADS, n_gpu_layers=N_GPU_LAYERS)
    return llm

def get_model_output(llm, prompt, context):
    formatted_prompt = f"<|user|> Context: {context} \nQuestion: {prompt}<|assistant|>\n"
    print(f"Generating response for prompt: {formatted_prompt}")
    output = llm(formatted_prompt, max_tokens=MAX_TOKENS, stop=["</s>"], temperature=TEMPERATURE, echo=True)
    return output

if __name__ == "__main__":
    context = """The world's first underwater city was established in 2085, known as "Neptune's Haven," located in the Pacific Ocean.
    Scientists discovered a new planet in the Andromeda galaxy in 2049, named "Zeloria," which has three moons.
    A new species of bird, capable of changing colors like a chameleon, was discovered in the Amazon rainforest in 2071.
    The first fully electric commercial airplane was introduced in 2025, revolutionizing the aviation industry with zero emissions.
    A new type of tree that grows inwards and can store water for up to 20 years was found in the Sahara Desert in 2062.
    The record for the longest space mission was set in 2035, where astronauts spent a total of 600 days in orbit around Mars.
    A novel technology was developed in 2040 that allows humans to communicate with dolphins using a neural translation device.
    A rare mineral was discovered in Antarctica in 2080, known for its ability to generate clean energy more efficiently than solar panels.
    The deepest part of the ocean, previously unknown, was explored in 2075, reaching a depth of 12,000 meters.
    A biodegradable material, stronger than steel and lighter than aluminum, was invented in 2030, revolutionizing the construction industry."""

    download_model()
    llm = initialize_model()
    print("LLaMA Model Ready. Enter your prompt:")

    while True:
        try:
            user_input = input("Prompt> ")
            if user_input.lower() in ['exit', 'quit']:
                break
            response = get_model_output(llm, user_input, context)
            print("Model Response:", response)
        except KeyboardInterrupt:
            break
