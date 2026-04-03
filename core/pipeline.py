"""
👑 Pipeline Master
Responsibility: Connect all modules together.

Provides two main flows:
1. Data Ingestion Flow (Loader -> Cleaner -> Chunker -> Embedder -> Vector DB)
2. Q&A Flow (Query -> Retriever -> Generator -> Verifier -> Confidence)

ALIGNED: All modules now use Ollama (nomic-embed-text + llama3).
"""

from ingestion.loader import load_documents
from ingestion.cleaner import clean_text
from ingestion.chunker import split_into_chunks
from embeddings.embedder import get_embedding_function
from embeddings.vector_store import store_chunks, load_vector_db
from rag.retriever import retrieve_docs
from rag.generator import generate_answer
from verification.verifier import verify_answer
from verification.confidence import compute_confidence
from core.config import DB_DIR, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

def run_ingestion_pipeline():
    """Runs the full ingestion pipeline to build the vector DB."""
    print("1. Loading documents...")
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found in", DATA_DIR)
        return
        
    print("2. Cleaning text...")
    cleaned_docs = clean_text(docs)
    
    print("3. Splitting into chunks...")
    chunks = split_into_chunks(cleaned_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    print("4. Embedding and storing...")
    embed_func = get_embedding_function()
    store_chunks(chunks, embed_func, DB_DIR)
    print("Pipeline complete!")

def run_qa_pipeline(query: str):
    """
    Runs the Q&A generation pipeline.
    Returns:
        dict: containing 'answer', 'context', 'verified', 'confidence'.
    """
    embed_func = get_embedding_function()
    db = load_vector_db(embed_func, DB_DIR)
    
    if db is None:
        return {
            "answer": "Database not initialized. Please run ingestion first.",
            "context": [],
            "verified": False,
            "confidence": 0.0
        }

    context = retrieve_docs(query, db, top_k=TOP_K)
    
    if not context:
        return {
            "answer": "I could not find this information in the retrieved documents.",
            "context": [],
            "verified": True, 
            "confidence": 0.0
        }
    
    answer = generate_answer(query, context)
    is_verified = verify_answer(answer, context)
    confidence = compute_confidence(answer, context)
    
    return {
        "answer": answer,
        "context": context,
        "verified": is_verified,
        "confidence": confidence
    }
