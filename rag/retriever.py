# rag/retriever.py
"""
Person 5: Document Retriever
Responsibility: Retrieve relevant chunks from ChromaDB using Ollama embeddings.

ALIGNED: Uses the same Ollama nomic-embed-text model as build_db.py and app.py.
"""

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from core.config import DB_DIR, TOP_K, EMBEDDING_MODEL, COLLECTION_NAME


def retrieve_chunks(query, db=None, top_k=None):
    """
    Retrieve the most relevant document chunks for a given query.

    Args:
        query (str): The user's question.
        db (Chroma, optional): Pre-loaded vector store. If None, loads from disk.
        top_k (int, optional): Number of results to return. Defaults to config TOP_K.

    Returns:
        list: List of LangChain Document objects.
    """
    if top_k is None:
        top_k = TOP_K

    if db is None:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    docs = db.similarity_search(query, k=top_k)
    return docs


# Alias for pipeline.py compatibility
retrieve_docs = retrieve_chunks