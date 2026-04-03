import os
import re
import time
import streamlit as st
from rag_engine import RAGEngine
from feedback.logger import log_feedback
from feedback.analytics import summary as feedback_summary

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIT Verification Assistant 🎓",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Theme Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #0f172a;
    color: #e2e8f0;
}

/* Header Styling */
.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 30px 40px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 30px;
    text-align: center;
}
.core-idea {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 10px;
}

/* Custom Answer Card */
.answer-box {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    border-left: 5px solid #3b82f6;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* Citation Markers */
.citation {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #3b82f6;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    width: 16px;
    height: 16px;
    border-radius: 4px;
    margin-left: 4px;
    vertical-align: super;
}

/* Span Highlighting */
.span-supported {
    background: rgba(52, 211, 153, 0.15);
    border-bottom: 2px solid #10b981;
}
.span-unsupported {
    background: rgba(248, 113, 113, 0.15);
    border-bottom: 2px dashed #ef4444;
}

/* Evidence Highlight in Chunk */
.evidence-highlight {
    background: rgba(255, 255, 0, 0.2);
    font-weight: 600;
    color: #f8fafc;
    border-left: 3px solid #fbbf24;
    padding-left: 5px;
}

/* Source Card */
.source-card {
    background: #1e293b;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
}
.source-tag {
    font-size: 0.7rem;
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    padding: 2px 10px;
    border-radius: 4px;
    border: 1px solid rgba(59, 130, 246, 0.3);
    margin-right: 5px;
}

.citation-badge {
    display: inline-block;
    background: #3b82f6;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 4px;
    margin-left: 5px;
    vertical-align: middle;
    cursor: help;
}

</style>
""", unsafe_allow_html=True)

# ── Engine Initialization ─────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    return RAGEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Helper Functions ──────────────────────────────────────────────────────────
def render_verified_answer(verification: dict):
    # The 'cited_answer' already contains [Source | Page | Section] markers
    answer_text = verification.get("cited_answer", "")
    
    # regex to find [Source | pg.X | Section] and wrap in styled span
    # Example: [CIT_Handbook.txt | pg.12 | General]
    styled_answer = re.sub(
        r'\[([^\]]+)\]', 
        r'<span class="citation-badge" title="Verified Source">\1</span>', 
        answer_text
    )
    
    html = f'<div style="line-height: 1.8; color: #cbd5e1;">{styled_answer}</div>'
    return html

def highlight_evidence_in_text(full_text: str, evidence_list: list[str]) -> str:
    highlighted = full_text
    for ev in evidence_list:
        pattern = re.compile(re.escape(ev), re.IGNORECASE)
        highlighted = pattern.sub(f'<span class="evidence-highlight">{ev}</span>', highlighted)
    return highlighted

# ── Sidebar & Knowledge Management ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #60a5fa;'>🎓 CIT Intelligence</h1>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 🖼️ Multi-Modal Upload (Step 14)")
    uploaded_files = st.file_uploader(
        "Upload Image or PDF", 
        type=["png", "jpg", "jpeg", "pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Synchronise Knowledge"):
            with st.spinner("🧠 Re-building semantic database..."):
                from core.config import DATA_DIR
                os.makedirs(DATA_DIR, exist_ok=True)
                
                for f in uploaded_files:
                    path = os.path.join(DATA_DIR, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                
                # Trigger the build process
                from build_db import build_db
                build_db()
                st.success("✅ Database Synchronised!")
                time.sleep(1)
                st.rerun()

    st.divider()
    st.markdown("### 📊 System Health")
    st.metric("Reasoning Model", "Ollama / Gemma 3", delta="High Fidelity")
    st.metric("Verification Mode", "Strict (70%)", delta="Active")
    st.caption("Advanced Metrics: Faithfulness / Precision / Relevance 📈")
    
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.rerun()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; color:#60a5fa;">CIT Verification Assistant v5.5</h1>
    <p class="core-idea">"Independently auditing every claim through research-grade metrics."</p>
</div>
""", unsafe_allow_html=True)

# ── Chat Logic ────────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f"**{msg['content']}**")
        else:
            res = msg.get("result", {})
            verif = res.get("verification", {})
            metrics = res.get("metrics", {})
            
            t1, t2, t3, t4 = st.tabs(["✨ Verified Response", "📜 Source Evidence", "📦 Retrieved Chunks", "📊 Advanced Analytics"])
            
            with t1:
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown(render_verified_answer(verif), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("🤔 Reasoning & Confidence Trace"):
                    conf = float(res.get("confidence", 0.0))
                    st.write(f"**Intent Detected:** `{res.get('intent', 'Query')}`")
                    st.write(f"**Overall Hybrid Confidence: {int(conf*100)}%**")
                    st.progress(conf)
                    
                    st.markdown("---")
                    st.markdown("#### 🔬 Metric Breakdown")
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Rerank Score (BGE):** `{metrics.get('rerank_score', 0)}`")
                    c2.markdown(f"**Embedding Sim:** `{metrics.get('sim_score', 0)}`")
                    
                    if res.get("intent") in ["COMPARISON", "COMPOSITE", "AGGREGATION"]:
                        st.success("🔄 **Multi-Hop Synthesis Active** — Reasoned across multiple thematic clusters.")

            with t2:
                st.info("Direct sentence-level grounding. Yellow highlights show exact evidence found in your documents.")
                supp_sent = verif.get("supported_sentences", [])
                for idx, doc in enumerate(res.get("docs", [])):
                    ev_this_doc = [s["text"] for s in supp_sent if s["doc_idx"] == idx]
                    st.markdown(f"""
                    <div class="source-card">
                        <div style="margin-bottom: 8px;">
                            <span class="source-tag">SOURCE [{idx+1}]</span>
                            <span class="source-tag">{doc.metadata.get('source', 'Unknown')}</span>
                            <span class="source-tag">PAGE {doc.metadata.get('page', '?')}</span>
                        </div>
                        <div style="font-size: 0.95rem; color: #cbd5e1;">{highlight_evidence_in_text(doc.page_content, ev_this_doc)}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with t3:
                st.info("Raw blocks extracted from ChromaDB before LLM synthesis.")
                for idx, doc in enumerate(res.get("docs", [])):
                    with st.expander(f"Chunk [{idx+1}] | Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}"):
                        st.code(doc.page_content, language="text")
                        st.json(doc.metadata)

            with t4:
                st.markdown("### 📈 Verification Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Faithfulness", f"{int(verif.get('score', 0)*100)}%", help="Precision of the answer relative to the context.")
                m2.metric("Context Precision", f"{int(metrics.get('context_precision', 0)*100)}%", help="How relevant the retrieved chunks are to the query.")
                m3.metric("Answer Relevance", f"{int(metrics.get('answer_relevance', 0)*100)}%", help="How well the final answer addresses the user's intent.")
                
                st.divider()
                st.markdown("### ⏱️ Performance")
                st.write(f"**Total Latency:** {metrics.get('latency', 0)}s")
                st.write(f"**Rerank Signal:** {metrics.get('rerank_score', 0)}")

            followups = res.get("followups", [])
            if followups:
                st.write("---")
                cols = st.columns(len(followups))
                for j, q in enumerate(followups):
                    if cols[j].button(q, key=f"fup_{i}_{j}", use_container_width=True):
                        st.session_state.pending_query = q
                        st.rerun()

if "pending_query" in st.session_state:
    p = st.session_state.pop("pending_query")
    st.session_state.messages.append({"role": "user", "content": p})
    st.rerun()

if prompt := st.chat_input("Ask about CIT rules..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(f"**{prompt}**")
    
    with st.chat_message("assistant"):
        engine = get_engine()
        with st.spinner("🛡️ Calculating hybrid confidence scores..."):
            res = engine.query(prompt)
            st.session_state.messages.append({"role": "assistant", "content": res["answer"], "result": res})
            st.rerun()
