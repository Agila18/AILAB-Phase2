# End-to-End RAG Student Assistant

## Project Structure & Team Assignments

This project is divided carefully to ensure **no overlap**, **clear boundaries**, and **easy integration**.

- `app.py`: UI (Agila & Person 1)
- `core/`: Config & Pipeline integration (Agila)
- `ingestion/`: Loading, cleaning, and chunking (Person 1 & 2)
- `embeddings/`: Embeddings & Vector Store (Person 3 & 4)
- `rag/`: Retrieval, Prompting & LLM Generation (Person 5)
- `verification/`: Answer verification & Confidence scoring (Person 6)
- `data/`: Put your PDF/DOCX/TXT files here
- `db/`: Chroma database directory

## 🚨 INTERFACE CONTRACT (CRITICAL)
Everyone MUST use this exact data format whenever communicating chunks/context:
```python
[
  {
    "text": "Extracted text content...",
    "source": "filename.pdf",
    "page": 1 
  }
]
```

## Setup Instructions
1. **Install Ollama**: Download and install [Ollama](https://ollama.com/) on your local machine.
2. **Pull Required Models**: Open your terminal and pull the LLM and Embedding models:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```
3. **Install dependencies**: `pip install -r requirements.txt`
4. Place your documents inside the `data/` folder.
5. Run the application: `streamlit run app.py`

## Team Instructions
- Build your module *independently*. 
- **Follow the exact dict format.**
- Do not rename functions.
- Test your module alone before merging.
- **Innovate!** Add logging, fallback responses, or advanced methods within your files as long as the inputs and outputs remain the same.
