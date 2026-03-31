"""
Person 3: Embeddings Generator
Responsibility: Convert text chunks into vector embeddings.

Provides a factory function that returns an OllamaEmbeddings object
for use by the vector store and retriever.

Requires Ollama running locally with nomic-embed-text pulled.
"""

from langchain_ollama import OllamaEmbeddings
from core.config import EMBEDDING_MODEL


def get_embedding_function():
    """
    Returns the embedding function to be used by the Vector DB.

    Returns:
        OllamaEmbeddings: Configured embedding object using nomic-embed-text.
    """
    return OllamaEmbeddings(model=EMBEDDING_MODEL)
