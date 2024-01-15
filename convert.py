import logging
from tqdm import tqdm
import os
import time
import chromadb
import textwrap
import subprocess
from langchain_community.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings

# MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# MODEL_NAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_REPO = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_NAME = "llama-2-7b-chat.Q4_K_M.gguf"
LOCAL_DIR = "/models"
CHORMADB_PATH = 'chroma_db_data'
CHORMADB_COLLECTION_NAME = 'web_pages'

model_config = {
    "n_ctx": 4096,
    "n_threads": 10,
    "n_gpu_layers": 0,
    "max_tokens": 256,
    "temperature": 0.1
}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def convert_pages_to_chromadb():
    chroma_client = chromadb.PersistentClient(path=CHORMADB_PATH)
    collection = chroma_client.get_or_create_collection(name=CHORMADB_COLLECTION_NAME)

    total_files = sum([len(files) for r, d, files in os.walk("pages") if files])
    processed_files = 0

    for root, _, files in os.walk("pages"):
        for file in files:
            if file.endswith(".html"):
                start_time = time.time()
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                chunks = textwrap.wrap(content.strip(), 1000)

                with tqdm(total=len(chunks), desc=f"Embedding file {file}") as pbar:
                    embeddings = generate_embeddings(chunks, pbar)
                chunk_ids = [f"{file}_chunk_{index}" for index in range(len(chunks))]
                collection.upsert(
                    documents=chunks,
                    ids=chunk_ids,
                    embeddings=embeddings
                )

                end_time = time.time()
                processed_files += 1
                print(f"######## Processed {file}. {processed_files}/{total_files} files done. Time taken: {end_time - start_time:.2f} seconds.")

def generate_embeddings(chunks, pbar):
    embedding_function = LlamaCppEmbeddings(model_path=os.path.join(LOCAL_DIR, MODEL_NAME))
    embeddings = []

    for chunk in chunks:
        start_time = time.time()
        embedding = embedding_function.embed_query(chunk)
        embeddings.append(embedding)
        end_time = time.time()
        pbar.update()
        print(f"######## Embedding successful. Time taken: {end_time - start_time:.2f} seconds. chunk: {chunk}")

    return embeddings
    
    
download_model()
initialize_model()
    
convert_pages_to_chromadb()