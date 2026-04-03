import os
import streamlit as st
from rag_engine import RAGEngine
from verification.verifier import verify_answer
from verification.confidence import compute_confidence

# --- Page Configuration ---
st.set_page_config(
    page_title="CIT Assistant 🚀",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def get_engine():
    return RAGEngine()

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .answer-card {
        background: #1e293b;
        padding: 24px;
        border-radius: 12px;
        border-left: 6px solid #3b82f6;
        margin-bottom: 20px;
    }
    .user-bubble {
        background: #334155;
        padding: 12px 18px;
        border-radius: 15px 15px 0 15px;
        margin-left: auto;
        max-width: 80%;
        margin-bottom: 20px;
    }
    .confidence-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 CIT Student Assistant")
st.markdown("Precision RAG enabled with Cross-Encoder Reranking.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="answer-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="font-weight:bold; font-size:1.1rem;">✅ Answer</span>
                <span class="confidence-badge" style="background:{'#22c55e' if msg['confidence'] > 0.7 else '#eab308'};">
                    Confidence: {int(msg['confidence']*100)}%
                </span>
            </div>
            {msg['content']}
            <div style="margin-top:15px; font-size:0.8rem; color:#94a3b8;">
                Verified: {'✔️' if msg['verified'] else '⚠️'}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Input
query = st.chat_input("Ask about HOD, attendance, scholarship...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    
    engine = get_engine()
    
    with st.spinner("🧠 Thinking with High-Accuracy Engine..."):
        answer, context = engine.query(query)
        is_verified = verify_answer(answer, context)
        conf = compute_confidence(answer, context)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "verified": is_verified,
            "confidence": conf
        })
    st.rerun()
