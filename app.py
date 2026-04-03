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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Main App Container */
.stApp {
    background: #f8fafc;
    color: #0f172a;
}

/* Global Font Settings */
* { font-family: 'Inter', sans-serif; }

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Main Header Card */
.main-header {
    background: #ffffff;
    padding: 2.5rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.core-idea {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Verified Answer Box */
.answer-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    border-left: 6px solid #2563eb;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    color: #1e293b;
}

/* Citation Badges (In-text) */
.citation-badge {
    display: inline-flex;
    align-items: center;
    background: #dbeafe;
    color: #1d4ed8;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    margin: 0 4px;
    border: 1px solid #bfdbfe;
    vertical-align: middle;
    cursor: help;
}

/* Evidence Highlighting within Documents */
.evidence-highlight {
    background: #fdf6b2; /* Soft yellow */
    color: #723b13;     /* Darker brown for contrast */
    font-weight: 600;
    padding: 0 2px;
    border-radius: 2px;
}

/* Source Evidence Cards */
.source-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.source-tag {
    font-size: 0.75rem;
    background: #f1f5f9;
    color: #475569;
    padding: 3px 10px;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    margin-right: 0.5rem;
    font-weight: 600;
}

/* Buttons & Interactive Elements */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}

/* Expander Styling */
.stExpander {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)

# ── Engine Initialization ─────────────────────────────────────────────────────
@st.cache_resource
def get_engine(v=1):
    return RAGEngine()

def get_engine_instance():
    return get_engine(v=2) # Incremented to bust stale cache

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Helper Functions ──────────────────────────────────────────────────────────
def render_verified_answer(verification: dict):
    answer_text = verification.get("cited_answer", "")
    
    # regex to find [Source | pg.X | Section] and wrap in styled span
    styled_answer = re.sub(
        r'\[([^\]]+)\]', 
        r'<span class="citation-badge" title="Verified Source">\1</span>', 
        answer_text
    )
    
    html = f'<div style="line-height: 1.8; color: #1e293b; font-size: 1.05rem;">{styled_answer}</div>'
    return html

def highlight_evidence_in_text(full_text: str, evidence_list: list[str]) -> str:
    highlighted = full_text
    for ev in evidence_list:
        pattern = re.compile(re.escape(ev), re.IGNORECASE)
        highlighted = pattern.sub(f'<span class="evidence-highlight">{ev}</span>', highlighted)
    return highlighted

# ── Sidebar & Knowledge Management ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🎓 CIT Intelligence</h1>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 🖼️ Multi-Modal Upload")
    uploaded_files = st.file_uploader(
        "Upload Image or PDF", 
        type=["png", "jpg", "jpeg", "pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Synchronise Knowledge", use_container_width=True):
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
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; color:#1e3a8a; font-weight: 800; font-size: 2.5rem;">CIT Verification Assistant</h1>
    <p class="core-idea">"Independently auditing every claim through research-grade metrics."</p>
</div>
""", unsafe_allow_html=True)

# ── Chat Logic ────────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f"**{msg['content']}**")
        else:
            res      = msg.get("result", {})
            verif    = res.get("verification", {})
            metrics  = res.get("metrics", {})
            conf     = float(res.get("confidence", 0.0))
            docs     = res.get("docs", [])
            supp_sent = verif.get("supported_sentences", [])

            # ── 1. ANSWER ───────────────────────────────────────────────────
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(render_verified_answer(verif), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── 2. CONFIDENCE SCORE ─────────────────────────────────────────
            conf_pct = int(conf * 100)
            if conf_pct >= 75:
                bar_color, conf_label = "#22c55e", "✅ High Confidence"
            elif conf_pct >= 50:
                bar_color, conf_label = "#f59e0b", "⚠️ Moderate Confidence"
            else:
                bar_color, conf_label = "#ef4444", "❌ Low Confidence"

            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1rem 1.25rem;margin-bottom:1rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-weight:700;color:#1e293b;font-size:0.95rem;">🎯 Confidence Score</span>
                <span style="font-weight:800;font-size:1.1rem;color:{bar_color};">{conf_pct}% &nbsp;·&nbsp; {conf_label}</span>
              </div>
              <div style="background:#e2e8f0;border-radius:999px;height:10px;overflow:hidden;">
                <div style="width:{conf_pct}%;background:{bar_color};height:100%;border-radius:999px;"></div>
              </div>
              <div style="display:flex;gap:1.5rem;margin-top:0.6rem;font-size:0.82rem;color:#64748b;flex-wrap:wrap;">
                <span>🔁 Rerank: <b>{metrics.get('rerank_score', 0)}</b></span>
                <span>🧬 Embed Sim: <b>{metrics.get('sim_score', 0)}</b></span>
                <span>📐 Context Precision: <b>{int(metrics.get('context_precision', 0)*100)}%</b></span>
                <span>🎯 Faithfulness: <b>{int(verif.get('score', 0)*100)}%</b></span>
                <span>⏱️ {metrics.get('latency', 0)}s</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── 3. CITATIONS / SOURCE EVIDENCE ──────────────────────────────
            if docs:
                with st.expander(f"📜 Source Citations — {len(docs)} document(s) retrieved", expanded=True):
                    st.caption("🟡 Yellow highlights = exact sentences grounding the answer.")
                    for idx, doc in enumerate(docs):
                        ev_this_doc = [s["text"] for s in supp_sent if s["doc_idx"] == idx]
                        src     = doc.metadata.get("source", "Unknown")
                        pg      = doc.metadata.get("page", "?")
                        section = doc.metadata.get("section", "")
                        highlighted   = highlight_evidence_in_text(doc.page_content, ev_this_doc)
                        badge_matched = "🟢 Evidence Found" if ev_this_doc else "⚪ No direct match"
                        st.markdown(f"""
                        <div class="source-card">
                          <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;flex-wrap:wrap;">
                            <span class="source-tag">SOURCE [{idx+1}]</span>
                            <span class="source-tag">📄 {src}</span>
                            <span class="source-tag">pg. {pg}</span>
                            {"<span class='source-tag'>" + section + "</span>" if section else ""}
                            <span style="margin-left:auto;font-size:0.78rem;font-weight:600;color:#16a34a;">{badge_matched}</span>
                          </div>
                          <div style="font-size:0.93rem;color:#334155;line-height:1.7;">{highlighted}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # ── 4. RAW CHUNKS ────────────────────────────────────────────────
            if docs:
                with st.expander(f"📦 Raw Retrieved Chunks ({len(docs)} blocks)", expanded=False):
                    st.caption("Unprocessed text blocks from ChromaDB before LLM synthesis.")
                    for idx, doc in enumerate(docs):
                        src = doc.metadata.get("source", "?")
                        pg  = doc.metadata.get("page", "?")
                        with st.expander(f"Chunk [{idx+1}]  ·  {src}  ·  pg. {pg}"):
                            st.code(doc.page_content, language="text")
                            st.json({k: v for k, v in doc.metadata.items()})

            # ── 5. ADVANCED ANALYTICS ────────────────────────────────────────
            with st.expander("📊 Advanced Analytics", expanded=False):
                m1, m2, m3 = st.columns(3)
                m1.metric("Faithfulness",      f"{int(verif.get('score', 0)*100)}%",               help="Precision relative to context")
                m2.metric("Context Precision", f"{int(metrics.get('context_precision', 0)*100)}%",  help="Relevance of retrieved chunks")
                m3.metric("Answer Relevance",  f"{int(metrics.get('answer_relevance', 0)*100)}%",   help="How well answer addresses the query")
                st.divider()
                st.write(f"**Intent:** `{res.get('intent', 'QUERY')}`")
                if res.get("intent") in ["COMPARISON", "COMPOSITE", "AGGREGATION"]:
                    st.success("🔄 Multi-Hop Synthesis active — reasoned across multiple clusters.")
                st.write(f"**Rerank Score:** `{metrics.get('rerank_score', 0)}`")
                st.write(f"**Embedding Similarity:** `{metrics.get('sim_score', 0)}`")
                st.write(f"**Latency:** `{metrics.get('latency', 0)}s`")
                if res.get("sources"):
                    st.write("**Sources:** " + ", ".join(f"`{s}`" for s in res["sources"]))

            # ── 6. FOLLOW-UP SUGGESTIONS ────────────────────────────────────
            followups = res.get("followups", [])
            if followups:
                cols = st.columns(len(followups))
                for j, q in enumerate(followups):
                    if cols[j].button(q, key=f"fup_{i}_{j}", use_container_width=True):
                        st.session_state.pending_query = q
                        st.rerun()

# ── Trigger Logic ─────────────────────────────────────────────────────────────
active_prompt = None

if "pending_query" in st.session_state:
    active_prompt = st.session_state.pop("pending_query")
    st.session_state.messages.append({"role": "user", "content": active_prompt})

if prompt := st.chat_input("Ask about CIT rules, fees, hostel, exams..."):
    active_prompt = prompt
    st.session_state.messages.append({"role": "user", "content": active_prompt})

if active_prompt:
    with st.chat_message("user"):
        st.markdown(f"**{active_prompt}**")
    
    with st.chat_message("assistant"):
        engine = get_engine_instance()
        
        # 🟢 New: Hybrid Streaming UI Logic
        full_answer = ""
        result = {}
        
        with st.status("🚀 Citations audit in progress...", expanded=True) as status:
            status_text = st.empty()
            answer_placeholder = st.empty()
            
            # Start the streaming generator
            for kind, data in engine.query_with_streaming(active_prompt):
                if kind == "status":
                    status_text.write(f"**Step:** {data}")
                    status.update(label=f"🚀 {data}")
                
                elif kind == "token":
                    full_answer += data
                    # Show partial answer to user for immediate feedback
                    answer_placeholder.markdown(full_answer + "▌")
                
                elif kind == "result":
                    result = data
                    
            status.update(label="✅ Citation Audit Complete", state="complete", expanded=False)
            
            # Final UI Refresh: Replace the raw stream with the fully verified/cited answer
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result.get("answer", full_answer), 
                "result": result
            })
            st.rerun()
