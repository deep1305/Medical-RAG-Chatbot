import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ollama Configuration
USE_OLLAMA = os.environ.get("USE_OLLAMA", "false").lower() == "true"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "all-minilm"  # For embeddings - smallest and fastest (45MB)
OLLAMA_LLM_MODEL = "qwen3-vl:30b-a3b-instruct"  # For LLM/chat

# HuggingFace Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Vector Store Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"

# Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50