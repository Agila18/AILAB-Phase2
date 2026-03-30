"""
👨‍💻 Person 4: Vector Store (ChromaDB) (🥈 SECOND MOST IMPORTANT)
Responsibility: Store chunks + metadata, and load the DB.

⚠️ WHY THIS IS CRITICAL:
If the retrieval DB is bad or improperly formatted -> Person 5 can't get correct data!

CRITICAL REQUIREMENT:
Takes the formatted chunks (text, source, page) and stories them in ChromaDB.
Expose a function to load/get the DB.

INNOVATION OPPORTUNITIES:
- Implement checks to avoid re-embedding existing documents (persistent DB hashing).
- Allow metadata filtering capabilities.
- Add try/except block robust enough to handle DB permission errors.
"""

# from langchain_chroma import Chroma
# from langchain.schema import Document
import os

def store_chunks(chunks: list[dict], embedding_function, persist_directory: str):
    """
    Creates or updates the Chroma vector database with the given chunks.
    
    Args:
        chunks (list[dict]): The chunked documents.
        embedding_function: The embedding function from embedder.py.
        persist_directory (str): Where to save the DB.
        
    Returns:
        vector_db: The initialized Chroma database object.
    """
    # TODO: Convert chunks format into standard LangChain Document objects internally needed for Chroma.
    # doc_objects = [Document(page_content=c['text'], metadata={"source": c['source'], "page": c['page']}) for c in chunks]
    
    # TODO: Initialize and persist Chroma DB.
    pass

def load_vector_db(embedding_function, persist_directory: str):
    """
    Loads the persistent Chroma vector database.
    """
    # TODO: Return the loaded Chroma DB.
    pass
