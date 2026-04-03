import os
import pickle
import shutil
from ingestion.loader import load_documents
from ingestion.cleaner import clean_text
from ingestion.chunker import split_into_chunks
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from core.config import DATA_DIR, DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# Where to save the BM25 corpus snapshot
BM25_CACHE = "bm25_docs.pkl"


def build_db():
    print("🚀 Starting Unified Ingestion Pipeline...")

    # ── Step 1: Load ────────────────────────────────────────────────────────────
    print(f"📂 Loading documents from {DATA_DIR}...")
    raw_docs = load_documents(DATA_DIR)
    if not raw_docs:
        print("❌ No documents found.  Please add files to data/.")
        return
    print(f"✅ Loaded {len(raw_docs)} raw files.")

    # ── Step 2: Clean ───────────────────────────────────────────────────────────
    print("🧹 Cleaning text...")
    cleaned_docs = clean_text(raw_docs)

    # ── Step 3: Chunk (metadata-aware) ──────────────────────────────────────────
    print("✂️  Splitting into metadata-rich chunks...")
    chunks = split_into_chunks(cleaned_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"✅ Created {len(chunks)} chunks.")

    # ── Step 4: Build LangChain Document objects ─────────────────────────────
    doc_objects = [
        Document(
            page_content=c["text"],
            metadata={
                "source":     c["source"],
                "page":       c["page"],
                "section":    c.get("section",    "General"),
                "department": c.get("department", "CIT"),
                "keywords":   c.get("keywords",   ""),
            },
        )
        for c in chunks
    ]

    # ── Step 5: Persist full-corpus BM25 snapshot ──────────────────────────────
    # Fix: Save ALL docs to pickle so BM25 is built on the ENTIRE corpus
    print(f"💾 Saving full BM25 corpus snapshot to {BM25_CACHE}...")
    with open(BM25_CACHE, "wb") as f:
        pickle.dump(doc_objects, f)
    print(f"✅ Saved {len(doc_objects)} docs to {BM25_CACHE}.")

    # ── Step 6: Embed & store in Chroma ─────────────────────────────────────
    print(f"🧠 Initialising Ollama embeddings ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(DB_DIR):
        print(f"🗑️  Clearing old DB at {DB_DIR}...")
        try:
            shutil.rmtree(DB_DIR)
        except Exception as e:
            print(f"⚠️  Could not delete old DB: {e}")

    print(f"📦 Creating Chroma vector store at {DB_DIR}...")
    Chroma.from_documents(
        documents=doc_objects,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    print("✨ SUCCESS — database is high-accuracy ready!")


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created '{DATA_DIR}'. Add your files, then run again.")
    else:
        build_db()
