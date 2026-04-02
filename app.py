import re
import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from hybrid_retriever import hybrid_search, build_bm25

# --- Page Configuration ---
st.set_page_config(
    page_title="CIT Coimbatore Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Environment Config ---
DB_DIR = "db"
COLLECTION_NAME = "student_docs"
EMBEDDING_MODEL = "nomic-embed-text"
TOP_K = 8

# --- Query Expansion (STEP 5) ---
def expand_query(query):
    """Expand abbreviations and acronyms for better retrieval."""
    query = query.lower()

    replacements = {
        "ai & ds": "artificial intelligence and data science",
        "ai ds": "artificial intelligence and data science",
        "ai&ds": "artificial intelligence and data science",
        "aids": "artificial intelligence and data science",
        "ai & ml": "artificial intelligence and machine learning",
        "ai ml": "artificial intelligence and machine learning",
        "hod": "head of department",
        "cse": "computer science and engineering",
        "ece": "electronics and communication engineering",
        "eee": "electrical and electronics engineering",
        "it dept": "information technology",
        "mca": "master of computer applications",
        "mba": "master of business administration",
        "cgpa": "cumulative grade point average",
        "sgpa": "semester grade point average",
        "cbcs": "choice based credit system",
    }

    for key, value in replacements.items():
        if key in query:
            query += " " + value

    return query

# --- Cached Resources ---
@st.cache_resource
def get_vectorstore():
    if not os.path.exists(DB_DIR):
        return None
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return vectorstore

@st.cache_resource
def get_llm():
    return OllamaLLM(model="llama3", temperature=0)

@st.cache_resource
def get_bm25_index():
    """Build BM25 index from all documents in the vector store."""
    vs = get_vectorstore()
    if vs is None:
        return None, None
    all_docs = vs.similarity_search("", k=500)
    if not all_docs:
        return None, None
    bm25, docs = build_bm25(all_docs)
    return bm25, docs

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🎓 CIT Coimbatore Assistant")
    st.divider()
    
    st.markdown("#### Status")
    vs = get_vectorstore()
    doc_count = vs._collection.count() if vs else 0
    
    st.markdown(f"""
    <div class="status-item">
        <div class="status-label">📄 Documents</div>
        <div class="status-value">{doc_count}</div>
    </div>
    <div class="status-item">
        <div class="status-label">✨ Embeddings</div>
        <div class="status-value">{'✅ Ready' if vs else '❌ Missing'}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("#### Mode")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", False))
    
    st.divider()
    st.caption("v2.0.0 | Hybrid Search + Query Expansion")

# --- Dynamic Theme Calculations ---
bg_color = "#0f172a" if st.session_state.dark_mode else "#f8fafc"
card_bg = "#1e293b" if st.session_state.dark_mode else "#ffffff"
text_color = "#f8fafc" if st.session_state.dark_mode else "#1e293b"
secondary_text = "#94a3b8" if st.session_state.dark_mode else "#64748b"
border_color = "#334155" if st.session_state.dark_mode else "#e2e8f0"

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_color}; font-family: 'Inter', sans-serif; }}
    
    section[data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        border-right: 1px solid {border_color};
    }}
    
    [data-testid="stSidebar"] *, .stMarkdown {{ color: {text_color} !important; }}

    .answer-card {{
        background: {card_bg};
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border-left: 6px solid #3b82f6;
        margin-bottom: 20px;
        color: {text_color} !important;
    }}

    .user-bubble {{
        background-color: {card_bg};
        color: {text_color} !important;
        border: 1px solid {border_color};
        padding: 12px 18px;
        border-radius: 15px 15px 0 15px;
        margin-left: auto;
        max-width: 80%;
        margin-bottom: 20px;
    }}

    .status-item {{
        background: {bg_color};
        border: 1px solid {border_color};
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
    }}

    .source-snippet {{
        background: {card_bg};
        padding: 12px;
        border-radius: 6px;
        border: 1px solid {border_color};
        font-size: 0.9rem;
    }}
    
    [data-testid="stChatInput"] textarea {{ 
        background-color: {card_bg} !important; 
        color: {text_color} !important; 
        caret-color: {text_color} !important;
        transition: all 0.2s ease-in-out;
    }}
    
    [data-testid="stChatInput"] textarea:focus {{
        border: 2px solid #3b82f6 !important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.6) !important;
        background-color: {bg_color} !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- Main Interface ---
st.title("Student Policy Assistant")
st.markdown("Grounded in official CIT Coimbatore documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-bubble">You: {message["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f"""
            <div class="answer-card">
                <div style="font-weight:bold; font-size:1.2rem; margin-bottom:10px;">✅ {message["title"]}</div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.markdown(f"**📄 {src['file']}**\n\n> {src['snippet']}...")

# Query Input
query = st.chat_input("Ask about attendance, scholarship, HOD, fees, etc.")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.rerun()

# RAG Logic Trigger
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    
    if vs:
        llm = get_llm()
        bm25, bm25_docs = get_bm25_index()
        
        # ✅ STEP 8: Multi-Query Handling — split compound queries
        sub_queries = [q.strip() for q in re.split(r'\band\b|,', user_query, flags=re.IGNORECASE) if q.strip()]
        if len(sub_queries) < 2:
            sub_queries = [user_query]

        # Load System Prompt
        system_prompt = "You are a helpful assistant."
        if os.path.exists("system_prompt.txt"):
            with open("system_prompt.txt", "r") as f:
                system_prompt = f.read().strip()

        # ✅ Per-sub-query retrieval + generation for compound queries
        if len(sub_queries) > 1:
            answers = []
            all_sources = []
            for sq in sub_queries:
                expanded_sq = expand_query(sq)
                if bm25 and bm25_docs:
                    results = hybrid_search(expanded_sq, vs, bm25, bm25_docs, k=TOP_K)
                else:
                    results = vs.similarity_search(expanded_sq, k=TOP_K)
                
                chunks = [res.page_content for res in results]
                all_sources.extend([{"file": res.metadata.get("source", "Unknown"), "snippet": res.page_content[:150]} for res in results])
                
                context_text = "\n\n---\n\n".join(chunks)
                prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {sq}\n\nAnswer:"
                
                try:
                    sub_answer = llm.invoke(prompt)
                    answers.append(sub_answer)
                except Exception as e:
                    answers.append(f"Error: {str(e)}")
            
            answer = "\n\n".join(answers)
            sources = all_sources
        else:
            # ✅ STEP 5: Expand query for single queries
            expanded_query = expand_query(user_query)
            
            # ✅ STEP 7: Hybrid Search (semantic + BM25)
            if bm25 and bm25_docs:
                all_results = hybrid_search(expanded_query, vs, bm25, bm25_docs, k=TOP_K)
            else:
                all_results = vs.similarity_search(expanded_query, k=TOP_K)
            
            chunks = [res.page_content for res in all_results]
            sources = [{"file": res.metadata.get("source", "Unknown"), "snippet": res.page_content[:150]} for res in all_results]
            
            context_text = "\n\n---\n\n".join(chunks)
            prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {user_query}\n\nAnswer:"
            
            try:
                with st.spinner("Analyzing documents..."):
                    answer = llm.invoke(prompt)
            except Exception as e:
                answer = f"Error communicating with LLM: {str(e)}"
            
        st.session_state.messages.append({
            "role": "assistant",
            "title": "Policy Retrieval Result",
            "content": answer,
            "sources": sources
        })
        st.rerun()
    else:
        st.error("Vector database not found! Please run `python build_db.py` first.")
