"""
👨‍💻 Person 3: Embeddings Generator
Responsibility: Convert text chunks into vector embeddings.

CRITICAL REQUIREMENT:
Provide a function that returns an embedding function supported by LangChain and ChromaDB.
🚨 REQUIRES OLLAMA: You must have Ollama installed and `nomic-embed-text` downloaded (`ollama pull nomic-embed-text`).

INNOVATION OPPORTUNITIES:
- Use local embeddings via Ollama for full offline support.
- Try different standard models (e.g., BAAI/bge-m3, sentence-transformers/all-MiniLM-L6-v2) for speed/accuracy.
- Add hardware acceleration (CUDA/Metal).
"""

# from langchain_community.embeddings import OllamaEmbeddings
# from core.config import EMBEDDING_MODEL

def get_embedding_function():
    """
    Returns the embedding function to be used by the Vector DB.
    """
    
    # TODO: Configure and return your chosen Ollama embedding sequence.
    # e.g., return OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    pass
