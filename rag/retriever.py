# rag/retriever.py

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

DB_PATH = "db"
TOP_K = 3

def retrieve_chunks(query):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    docs = db.similarity_search(query, k=TOP_K)

    return docs