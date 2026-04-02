import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

DATA_DIR = "data"
DB_DIR = "db"
COLLECTION_NAME = "student_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "nomic-embed-text"

def load_documents():
    documents = []
    # Load all txt files.
    for file in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        try:
            loader = TextLoader(file, encoding='utf-8')
            documents.extend(loader.load())
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Failed to load {file}: {e}")
            
    # Load PDFs if any
    for file in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        try:
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Failed to load PDF {file}: {e}")
            
    return documents

def build_db():
    print("Starting fast document ingestion...")
    raw_docs = load_documents()
    if not raw_docs:
        print("No documents found in 'data/' folder. Please add some files.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = text_splitter.split_documents(raw_docs)
    print(f"Split into {len(docs)} chunks.")

    print(f"Initializing Ollama embeddings ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    if os.path.exists(DB_DIR):
        print(f"Found existing DB at {DB_DIR}. Clearing it for fresh index...")
        try:
            shutil.rmtree(DB_DIR)
        except Exception as e:
            print("Could not delete old DB:", e)
        
    print(f"Creating vector store at {DB_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    )
    print("✅ Indexing complete! Database is ready.")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created '{DATA_DIR}'. Please add documents.")
    else:
        build_db()
