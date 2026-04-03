# 🎓 CIT Intelligence Assistant: Verified RAG System v4.5

> "An intelligent, verifiable, and explainable RAG system that answers using real documents with proof."

This is a production-grade **Retrieval-Augmented Generation (RAG)** assistant designed for college students. Unlike standard chatbots, this system **proves, verifies, and explains** every answer using official documentation.

---

## 🚀 Key Features

- **🧠 Intent-Aware Retrieval**: Automatically detects if a query is *Factual*, *Comparison*, or *Procedural* and adjusts its search strategy.
- **🛡️ Hallucination Killer**: Features a **Span-Level Verifier** that highlights grounded sentences in green and flags ungrounded claims.
- **🔍 Multi-Hop Reasoning**: Synthesizes answers by combining data from multiple separate documents.
- **📏 Advanced Metrics**: Real-time tracking of **Faithfulness** (grounding) and **Answer Relevance**.
- **💎 Premium UI**: Professional Streamlit interface with chat history, source citations, and performance analytics.
- **🔄 Feedback Loop**: Integrated 👍/👎 rating system with persistent logging.

---

## 📁 Project Structure

```bash
rag-assistant/
├── app.py              # 🚀 Main Entry Point (Streamlit UI)
├── rag_engine.py       # 🧠 Unified RAG Engine (Orchestration)
├── build_db.py         # 📦 Vector DB Ingestion Pipeline
├── requirements.txt    # 🐍 Python Dependencies
├── core/               # ⚙️ Global Configuration & Constants
├── data/               # 📄 Knowledge Base (PDFs/TXTs)
├── db/                 # 🗄️ Persistent Vector Database (ChromaDB)
├── feedback/           # 📊 User Feedback Logs (JSONL)
├── query/              # 🔍 Query Intelligence (Rewriter/Intent)
├── retrieval/          # 🐕 Advanced Retrieval (Hybrid/Multi-Hop)
└── verification/       # 🛡️ Verification Suite (Span-Level/Confidence)
```

---

## 🛠️ Step-by-Step Installation

### 1. Prerequisite: Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com/).

### 2. Clone the Repository
```bash
git clone https://github.com/Agila18/AILAB-Phase2.git
cd AILAB-Phase2/rag-assistant
```

### 3. Setup Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Pull AI Models
The system uses **Gemma-3 (1b)** for generation and **Nomic Embed Text** for vectors.
```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

---

## 🏃 Running the System

### 1. Build the Database
Place your PDF/TXT files in the `data/` folder and run the ingestion pipeline:
```bash
python build_db.py
```
This will create a persistent ChromaDB index in the `db/` folder.

### 2. Launch the Application
Start the Streamlit interface:
```bash
streamlit run app.py
```

---

## 📊 Analytics & Verification

- **Confidence Score (%)**: Based on keyword overlap and source density.
- **Faithfulness Score**: Measures the degree of grounding in the retrieved context.
- **Relevance Score**: Measures how well the AI addressed your specific question.
- **Latency (s)**: Real-time response time tracking.

## 🛡️ System Transparency
Every response is processed through a **Verification Gate**. If the confidence is below 30% or the info isn't in CIT documents, the AI will politely refuse to hallucinate, directing you to the college office instead.

---

## 🛠️ Built With
- **Ollama**: Local LLM Orchestration.
- **LangChain**: RAG Pipeline & Vector Processing.
- **ChromaDB**: Vector search backend.
- **Streamlit**: Material-design inspired UI.
- **Sentence-Transformers**: Cross-Encoder reranking.

---

> [!TIP]
> Use the **"Advanced Analytics"** tab in the UI during presentations to show judges the internal reasoning and faithfulness metrics of your project.
