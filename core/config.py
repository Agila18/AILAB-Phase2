"""
🔧 Global Configuration
Constants like chunk_size, top_k, model names.
"""

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
DB_DIR = "./db"
DATA_DIR = "./data"
MODEL_NAME = "llama3" # Primary local LLM using Ollama
EMBEDDING_MODEL = "nomic-embed-text" # Primary local embedding model using Ollama
