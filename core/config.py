"""
🔧 Global Configuration
Constants like chunk_size, top_k, model names.
All files in the project MUST import from here — single source of truth.
"""

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 8
DB_DIR = "./db"
DATA_DIR = "./data"
COLLECTION_NAME = "student_docs"
MODEL_NAME = "gemma3:1b"
# 🚀 PERFORMANCE FIX: Use local CPU embeddings to avoid Ollama model swapping
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
