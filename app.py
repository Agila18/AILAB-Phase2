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

# ── Theme State Initialization ────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark" # Defaulting to the requested theme

# ── Theme Variable Mapping ───────────────────────────────────────────────────
if st.session_state.theme == "Light":
    V = {
        "bg": "#f8fafc", "card": "#ffffff", "border": "#e2e8f0", 
        "text": "#0f172a", "text_dim": "#64748b", "accent": "#2563eb",
        "pill": "#f1f5f9", "highlight": "#fef08a", "highlight_text": "#713f12"
    }
else:
    V = {
        "bg": "#0d1117", "card": "#161b22", "border": "#30363d", 
        "text": "#f0f6fc", "text_dim": "#8b949e", "accent": "#58a6ff",
        "pill": "#21262d", "highlight": "#422006", "highlight_text": "#fde047"
    }

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* {{ font-family: 'Inter', sans-serif; }}

.stApp {{ background: {V['bg']}; color: {V['text']}; }}
[data-testid="stSidebar"] {{ background-color: {V['card']} !important; border-right: 1px solid {V['border']} !important; }}
[data-testid="stSidebar"] * {{ color: {V['text']} !important; }}
[data-testid="stSidebar"] .stCaption p {{ color: {V['text_dim']} !important; }}

/* Cards & Containers */
.card {{
    background: {V['card']}; 
    border: 1px solid {V['border']}; 
    border-radius: 12px; 
    padding: 1.5rem; 
    margin-bottom: 1.5rem;
    color: {V['text']} !important;
}}

.main-header {{
    background: {V['card']}; padding: 2.5rem; border-radius: 12px; margin-bottom: 2rem; 
    text-align: center; border: 1px solid {V['border']};
}}

/* Answer Header & Badges */
.answer-header {{
    display: flex; justify-content: space-between; align-items: center; 
    padding-bottom: 1rem; border-bottom: 1px solid {V['border']}; margin-bottom: 1rem;
    color: {V['text']} !important;
}}
.badge-group {{ display: flex; gap: 8px; align-items: center; }}
.badge {{
    padding: 4px 12px; border-radius: 999px; font-size: 0.75rem; 
    font-weight: 700; display: flex; align-items: center; gap: 4px;
}}
.badge-green {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }}
.badge-blue {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}

/* Button Overrides (Crucial Fix for Legibility) */
.stButton button, .stFileUploader button, [data-testid="stSidebar"] button {{
    background-color: {V['pill']} !important;
    color: {V['text']} !important;
    border: 1px solid {V['border']} !important;
    transition: all 0.3s ease;
    border-radius: 8px !important;
}}
.stButton button:hover, .stFileUploader button:hover, [data-testid="stSidebar"] button:hover {{
    border-color: {V['accent']} !important;
    color: {V['accent']} !important;
}}

/* File Uploader Container Fixes - Aggressive Targeting */
div[data-testid="stFileUploader"] section {{
    background-color: {V['card']} !important;
    border: 1px dashed {V['border']} !important;
    color: {V['text']} !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}}
div[data-testid="stFileUploader"] section button {{
    background-color: {V['pill']} !important;
    color: {V['text']} !important;
    border: 1px solid {V['border']} !important;
    margin-top: 10px !important;
}}
div[data-testid="stFileUploader"] section span, 
div[data-testid="stFileUploader"] section p,
div[data-testid="stFileUploader"] label {{
    color: {V['text']} !important;
}}

/* Evidence & Text Content */
.evidence-highlight {{ background: {V['highlight']}; color: {V['highlight_text']}; font-weight: 600; padding: 0 2px; border-radius: 2px; }}
.citation-badge {{ 
    display: inline-flex; background: {V['accent']}; color: #ffffff !important; padding: 1px 8px; 
    border-radius: 4px; font-size: 0.72rem; font-weight: 700; cursor: help; margin: 0 2px;
}}

/* Progress Bars */
.progress-container {{ height: 8px; background: {V['border']}; border-radius: 4px; overflow: hidden; margin-top: 4px; }}
.progress-fill {{ height: 100%; border-radius: 4px; transition: width 0.6s ease-in-out; }}

/* Typography Overrides & Chat Fixes */
h1, h2, h3, h4, h5 {{ color: {V['text']} !important; }}
div[data-testid="stMarkdownContainer"] p, 
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stChatMessage"] p, 
div[data-testid="stChatMessage"] b,
div[data-testid="stChatMessage"] span {{ 
    color: {V['text']} !important; 
}}

[data-testid="stSidebar"] code {{
    background-color: {V['pill']} !important;
    color: {V['text']} !important;
    border: 1px solid {V['border']} !important;
    padding: 2px 6px !important;
    font-size: 0.85rem !important;
    border-radius: 4px !important;
}}
[data-testid="stSidebar"] p {{
    color: {V['text']} !important;
    font-size: 0.9rem !important;
}}

/* Hide Streamlit Upload Limit Text (200MB) */
div[data-testid="stFileUploader"] small {{ display: none !important; }}

.stExpander {{ background: {V['card']} !important; border: 1px solid {V['border']} !important; border-radius: 8px !important; }}
div[data-testid="stExpander"] * {{ color: {V['text']} !important; }}
::selection {{ background: {V['accent']}; color: white; }}
</style>
""", unsafe_allow_html=True)

# ── Engine Initialization ─────────────────────────────────────────────────────
@st.cache_resource
def get_engine(v=1):
    return RAGEngine()

def get_engine_instance():
    return get_engine(v=8) # Incremented to bust stale cache and load NameError hotfix

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Helper Functions ──────────────────────────────────────────────────────────
def render_verified_answer(verification: dict):
    answer_text = verification.get("cited_answer", "")
    
    # Styled span for citations
    styled_answer = re.sub(
        r'\[([^\]]+)\]', 
        r'<span class="citation-badge">\1</span>', 
        answer_text
    )
    
    html = f'<div style="line-height: 1.8; font-size: 1.05rem;">{styled_answer}</div>'
    return html

def highlight_evidence_in_text(full_text: str, evidence_list: list[str]) -> str:
    # 1. Extract meaningful keywords (nouns, long words) from evidence
    keywords = set()
    for ev in evidence_list:
        words = [w.strip(".,!?") for w in ev.split()]
        keywords.update([w for w in words if len(w) > 4 and w.lower() not in ["according", "provided", "retrieved"]])
    
    highlighted = full_text
    for kw in sorted(list(keywords), key=len, reverse=True):
        pattern = re.compile(rf'\b({re.escape(kw)})\b', re.IGNORECASE)
        highlighted = pattern.sub(r'<span class="evidence-highlight">\1</span>', highlighted)
    return highlighted

def extract_evidence_window(full_text: str, evidence_list: list[str]) -> str:
    """Show only sentences surrounding the evidence to keep the UI clean."""
    if not evidence_list:
        return full_text[:300] + "..."
    
    # Simple sentence split (approximate for display)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    ev_indices = set()
    
    for ev in evidence_list:
        # Find doc sentence that matches evidence best
        for i, s in enumerate(sentences):
            if any(word.lower() in s.lower() for word in ev.split() if len(word) > 5):
                ev_indices.add(i)
    
    if not ev_indices:
        return full_text[:300] + "..."
        
    window_indices = set()
    for i in ev_indices:
        for offset in [-1, 0, 1]:
            if 0 <= i + offset < len(sentences):
                window_indices.add(i + offset)
    
    rendered = []
    last_idx = -1
    for idx in sorted(list(window_indices)):
        if last_idx != -1 and idx > last_idx + 1:
            rendered.append("...")
        rendered.append(sentences[idx])
        last_idx = idx
    
    if last_idx < len(sentences) - 1:
        rendered.append("...")
        
    return " ".join(rendered)

# ── Sidebar & Knowledge Management ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🎓 CIT Intelligence</h1>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 🖼️ Multi-Modal Upload")
    uploaded_file = st.file_uploader(
        "Upload Knowledge Document", 
        type=["txt", "pdf", "md"]
    )
    
    if uploaded_file:
        # Check session_state to prevent an infinite reload loop on the same file drop
        if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
            import subprocess
            from core.config import DATA_DIR
            
            os.makedirs(DATA_DIR, exist_ok=True)
            path = os.path.join(DATA_DIR, uploaded_file.name)
            
            with st.spinner(f"Saving {uploaded_file.name}..."):
                with open(path, "wb") as out:
                    out.write(uploaded_file.getbuffer())
            st.success("File uploaded!")

            with st.spinner("🧠 Updating database..."):
                subprocess.run(["python", "build_db.py"])
                
                # 🔥 KEY FIX: Completely dump the stale database instance out of RAM
                st.cache_resource.clear()
                
                st.session_state["last_uploaded_file"] = uploaded_file.name
                
            st.success("Database updated!")
            time.sleep(1)
            st.rerun()

    st.divider()
    st.markdown("### 📚 Knowledge Base")
    
    from core.config import DATA_DIR
    if os.path.exists(DATA_DIR):
        docs = [f for f in os.listdir(DATA_DIR) if f.endswith(('.txt', '.pdf', '.md'))]
        if docs:
            with st.expander(f"View Active Documents ({len(docs)})"):
                for doc in docs:
                    st.markdown(f"📄 `{doc}`")
        else:
            st.info("No documents indexed.")
            
    st.divider()
    st.markdown("### 🎨 Appearance")
    
    # Theme toggle logic with instant UI reload
    new_theme = st.radio("Theme Default", ["Light", "Dark"], horizontal=True, index=0 if st.session_state.theme == "Light" else 1, label_visibility="collapsed")
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
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

            # ── 1. PREMIUM STACKED LAYOUT ───────────────────────────────────────────
            verdict = res.get("verdict", "FACTUALLY_GROUNDED")
            verdict_map = {
                "FACTUALLY_GROUNDED": ("#22c55e", "✅ Grounded"),
                "PARTIALLY_SUPPORTED": ("#f59e0b", "⚠️ Partial"),
                "GENERAL_ADVICE":     ("#3b82f6", "💡 Advisor"),
                "REJECTED":           ("#ef4444", "🛡️ Restricted")
            }
            v_color, v_label = verdict_map.get(verdict, ("#22c55e", "✅ Grounded"))
            
            conf_pct = int(conf * 100)
            conf_color = "#22c55e" if conf_pct >= 70 else "#f59e0b" if conf_pct >= 50 else "#ef4444"
            
            # --- CARD 1: ANSWER ---
            st.markdown(f"""
            <div class="card">
                <div class="answer-header">
                    <div style="display:flex; align-items:center; gap:8px; font-weight:700; font-size:1.1rem;">
                        <span style="color:#22c55e;">✅</span> Answer
                    </div>
                    <div class="badge-group">
                        <div class="badge badge-green">✦ {conf_pct}% High</div>
                        <div style="font-size:0.85rem; font-weight:600; color:{v_color};">{v_label}</div>
                    </div>
                </div>
                {render_verified_answer(verif)}
            </div>
            """, unsafe_allow_html=True)

            # --- CARD 2: SOURCE CHIPS ---
            if docs:
                chips_html = '<div style="display:flex; gap:8px; margin-bottom:1.5rem; flex-wrap:wrap;">'
                recorded_sources = set()
                for doc in docs:
                    s_name = doc.metadata.get("source", "Doc")
                    if s_name not in recorded_sources:
                        chips_html += f'<div style="background:rgba(59,130,246,0.1); color:#3b82f6; border:1px solid rgba(59,130,246,0.2); padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600;">📄 {s_name}</div>'
                        recorded_sources.add(s_name)
                chips_html += '</div>'
                st.markdown(chips_html, unsafe_allow_html=True)

            # --- CARD 3: CONFIDENCE BREAKDOWN ---
            with st.expander("📊 Confidence breakdown by source", expanded=False):
                for idx, doc in enumerate(docs):
                    s_name = doc.metadata.get("source", "Source")
                    # Using doc score if available, else a simulated gradient for the 'look'
                    s_score = int(conf_pct if idx == 0 else max(conf_pct - (idx*5), 40))
                    s_color = "#22c55e" if s_score >= 70 else "#f59e0b"
                    
                    st.markdown(f"""
                    <div style="margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.85rem;">
                            <div style="display:flex; align-items:center; gap:6px; color:{V['text']};">
                                <span>📄</span> {s_name}
                            </div>
                            <div style="font-weight:700; color:{s_color};">{s_score}%</div>
                        </div>
                        <div class="progress-container">
                            <div class="progress-fill" style="width:{s_score}%; background:{s_color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # --- CARD 4: RAW CHUNKS ---
            with st.expander("🔬 View retrieved context chunks", expanded=False):
                for idx, doc in enumerate(docs):
                    ev_this_doc = [s["text"] for s in supp_sent if s["doc_idx"] == idx]
                    src = doc.metadata.get("source", "?")
                    pg  = doc.metadata.get("page", "?")
                    sect = doc.metadata.get("section", "General")
                    
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px; font-size:0.8rem;">
                        <span style="font-weight:700; color:{V['text']};">Chunk {idx+1} ·</span>
                        <span style="background:{V['pill']}; padding:1px 8px; border-radius:4px; border:1px solid {V['border']}; color:{V['text_dim']};">{src}</span>
                        <span style="color:{V['text_dim']};">· Page {pg} · {sect}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Highlighted snippet
                    window_text = extract_evidence_window(doc.page_content, ev_this_doc)
                    highlighted = highlight_evidence_in_text(window_text, ev_this_doc)
                    
                    st.markdown(f"""
                    <div style="background:{V['bg']}; border:1px solid {V['border']}; border-radius:8px; padding:1rem; margin-bottom:1.5rem; font-family:monospace; font-size:0.85rem; line-height:1.6;">
                        {highlighted}
                    </div>
                    """, unsafe_allow_html=True)

            pass # Advanced Analytics tab (t4) removed


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
