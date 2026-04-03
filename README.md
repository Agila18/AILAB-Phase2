# 🎓 CIT Intelligence Assistant: High-Fidelity RAG v5.0

> "A secure, explainable, and verifiable RAG system with fine-grained citation and multi-factor confidence scoring, sentence-level evidence mapping, and prompt injection defense to ensure reliability, transparency, and security in AI responses."

This is a production-ready **Retrieval-Augmented Generation (RAG)** assistant. It specializes in converting messy PDF/TXT documentation into structured, grounded, and secure intelligence for CIT students.

---

## 🚀 Key Features

*   **🛡️ Prompt Injection Defense**: Dynamic filtering of malicious instructions from documents and multi-layered system prompt hardening to guarantee security.
*   **📍 Sentence-Level Evidence Mapping**: Fine-grained citations where every generating sentence is mapped to a specific source document, page, and section.
*   **📏 Multi-Factor Confidence Scoring**: An industry-style hybrid formula (40% Rerank / 30% Vector / 20% Keyword / 10% Coverage) ensuring high reliability. Answers below 50% confidence trigger a strict safety gate.
- **👁️ Absolute Transparency**: "Level 1" UI features allow you to view the **Raw Retrieved Chunks** pulled from ChromaDB.
- **📊 Automated Benchmarking**: Full evaluation suite to measure Accuracy, Faithfulness, and Latency against a Gold Standard dataset.

---

## 📁 Project Structure

```bash
rag-assistant/
├── app.py              # 🚀 Main Entry Point (Premium Streamlit UI)
├── rag_engine.py       # 🧠 Unified RAG Engine (Orchestration & Security)
├── build_db.py         # 📦 Vector DB Ingestion Pipeline
├── core/               # ⚙️ Global Configuration
├── data/               # 📄 Knowledge Base (Place PDF/TXT here)
├── db/                 # 🗄️ Persistent ChromaDB Vector Store
├── evaluation/         # 📊 Automated Benchmark Suite (Accuracy/Faithfulness)
├── retrieval/          # 🐕 Advanced Retrieval (Hybrid BM25 + Vector)
└── verification/       # 🛡️ Grounding Suite (Span-Level Citations & Confidence)
```

---

## 🛠️ Installation & Setup

### 1. Prerequisite: Install Ollama
Download and install from [ollama.com](https://ollama.com/).

### 2. Setup Environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Pull Models
```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

### 4. Build Intelligence
Place your files in `data/` and run:
```bash
python build_db.py
```

---

## 📊 Evaluation & Benchmarking

The system includes a professional evaluation suite located in `evaluation/`.

**To run a full accuracy benchmark:**
```powershell
$env:PYTHONPATH="."
python evaluation/evaluate.py
python evaluation/summarize.py
```
This generates a detailed Pass/Fail report and calculates the project's **Accuracy Rate** and **Avg Faithfulness**.

---

## 🛡️ Built With
- **Ollama**: Local LLM Orchestration (Gemma 3).
- **ChromaDB**: High-performance vector database.
- **Sentence-Transformers**: Cross-Encoder (ms-marco) for RAG reranking.
- **Rank-BM25**: Hybrid lexical retrieval.
- **Streamlit**: Material-design inspired interactive UI.

---

> [!IMPORTANT]
> **Level 1 Compliance**: This project satisfies all assignment requirements including confidence scoring, raw chunk visibility, citation mapping, and faithfulness metrics.
