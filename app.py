import os
import streamlit as st
from rag_engine import RAGEngine
from feedback.logger import log_feedback
from feedback.analytics import summary as feedback_summary

st.set_page_config(
    page_title="CIT Student Assistant 🎓",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg,#0a0e1a 0%,#0f172a 50%,#0d1526 100%); color:#e2e8f0; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0f172a 0%,#1a1f35 100%); border-right:1px solid rgba(59,130,246,.2); }
.user-bubble { background:linear-gradient(135deg,#1e40af,#3b82f6); padding:14px 20px; border-radius:20px 20px 4px 20px; margin-left:auto; max-width:75%; margin-bottom:16px; font-size:.95rem; box-shadow:0 4px 15px rgba(59,130,246,.25); color:#fff; }
.answer-card { background:linear-gradient(135deg,rgba(30,41,59,.9),rgba(15,23,42,.95)); padding:24px 28px; border-radius:16px; border-left:4px solid #3b82f6; margin-bottom:8px; box-shadow:0 8px 32px rgba(0,0,0,.3); }
.badge-high   { background:#064e3b; color:#34d399; border:1px solid #34d399; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }
.badge-medium { background:#713f12; color:#fbbf24; border:1px solid #fbbf24; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }
.badge-low    { background:#7f1d1d; color:#f87171; border:1px solid #f87171; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }
.verify-ok   { color:#34d399; font-size:.82rem; }
.verify-warn { color:#fbbf24; font-size:.82rem; }
.rewrite-pill { display:inline-block; background:rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.4); color:#a5b4fc; padding:4px 14px; border-radius:20px; font-size:.78rem; margin-bottom:12px; }
.source-chip  { display:inline-block; background:rgba(59,130,246,.1); border:1px solid rgba(59,130,246,.3); color:#93c5fd; padding:2px 10px; border-radius:12px; font-size:.75rem; margin:2px 3px 2px 0; }
.unsupported-text { background:rgba(251,191,36,.12); border-left:3px solid #fbbf24; padding:4px 8px; border-radius:0 6px 6px 0; color:#fde68a; font-style:italic; margin:4px 0; display:block; font-size:.88rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    return RAGEngine()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_done" not in st.session_state:
    st.session_state.feedback_done = set()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px;">
        <div style="font-size:2.5rem;">🎓</div>
        <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">CIT Assistant</div>
        <div style="font-size:.75rem;color:#64748b;margin-top:4px;">
            Hybrid Search · Cross-Encoder · Multi-hop
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🔍 Metadata Filters")
    filter_source = st.selectbox(
        "Source Document",
        ["All Documents","attendance_rules.txt","scholarship.txt",
         "hostel_rules.txt","exam_policy.txt","placement_policy.txt",
         "fee_structure.txt","CIT_Academic_Calendar.txt"],
    )
    filter_section = st.text_input("Section (optional)", placeholder="e.g. Attendance Policy")

    metadata_filter = None
    fp = {}
    if filter_source != "All Documents":
        fp["source"] = filter_source
    if filter_section.strip():
        fp["section"] = filter_section.strip()
    if fp:
        metadata_filter = fp

    st.divider()
    st.markdown("### 💡 Quick Prompts")
    quick_prompts = [
        "What is the minimum attendance requirement?",
        "Who is the HOD of AI & DS?",
        "How do I apply for a scholarship?",
        "What are the hostel curfew timings?",
        "What happens if I have a backlog?",
        "What is the fee structure?",
    ]
    for p in quick_prompts:
        if st.button(p, key=f"qp_{p[:20]}", use_container_width=True):
            st.session_state["quick_prompt"] = p
    st.rerun() if "quick_prompt" in st.session_state and st.session_state.get("_qp_pending") else None

    st.divider()
    st.markdown("### 📊 Feedback Stats")
    try:
        stats = feedback_summary()
        if stats["total"] > 0:
            c1, c2 = st.columns(2)
            c1.metric("👍", stats["thumbs_up"])
            c2.metric("👎", stats["thumbs_down"])
            st.progress(stats["approval_rate"],
                        text=f"Approval: {int(stats['approval_rate']*100)}%")
            st.caption(f"Avg confidence: {stats['avg_confidence']:.0%} · Total: {stats['total']}")
        else:
            st.caption("No feedback yet.")
    except Exception:
        st.caption("No feedback log yet.")

    st.divider()
    st.markdown(
        "<div style='font-size:.7rem;color:#475569;text-align:center'>"
        "BM25 + Vector · Cross-Encoder · Multi-hop · Span Verifier</div>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:8px 0 24px;">
  <h1 style="margin:0;font-size:2rem;font-weight:700;
             background:linear-gradient(135deg,#60a5fa,#a78bfa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    🎓 CIT Student Assistant
  </h1>
  <p style="margin:6px 0 0;color:#64748b;font-size:.9rem;">
    Industry-grade RAG · Hybrid Search · Multi-hop · Span Verification
  </p>
</div>""", unsafe_allow_html=True)


def confidence_badge(conf: float) -> str:
    if conf >= 0.70:
        return f'<span class="badge-high">✦ {int(conf*100)}% High</span>'
    elif conf >= 0.45:
        return f'<span class="badge-medium">◈ {int(conf*100)}% Medium</span>'
    else:
        return f'<span class="badge-low">◇ {int(conf*100)}% Low</span>'


def render_message(idx: int, msg: dict):
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">💬 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        return

    result  = msg.get("result", {})
    conf    = result.get("confidence", 0.0)
    verif   = result.get("verification", {})
    rewrite = result.get("rewritten_query", "")
    sources = result.get("sources", [])
    answer  = msg["content"]

    if rewrite and rewrite.lower() != msg.get("original_query", "").lower():
        st.markdown(
            f'<div class="rewrite-pill">🔍 Interpreted as: <em>{rewrite}</em></div>',
            unsafe_allow_html=True,
        )

    verified_icon = (
        '<span class="verify-ok">✔ Grounded</span>'
        if verif.get("verified")
        else '<span class="verify-warn">⚠ Partially grounded</span>'
    )

    st.markdown(f"""
    <div class="answer-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
            <span style="font-weight:600;font-size:1rem;color:#e2e8f0;">✅ Answer</span>
            <div style="display:flex;gap:10px;align-items:center;">
                {confidence_badge(conf)}
                {verified_icon}
            </div>
        </div>
        <div style="color:#cbd5e1;line-height:1.7;font-size:.95rem;">{answer}</div>
    </div>""", unsafe_allow_html=True)

    unsupported = verif.get("unsupported_sentences", [])
    if unsupported:
        with st.expander(f"⚠️ {len(unsupported)} sentence(s) may not be fully grounded"):
            for sent in unsupported:
                st.markdown(f'<div class="unsupported-text">⚡ {sent}</div>',
                            unsafe_allow_html=True)
            st.caption("These sentences had low overlap with retrieved context.")

    if sources:
        chips = "".join(f'<span class="source-chip">📄 {s}</span>' for s in sources)
        st.markdown(f'<div style="margin:10px 0 4px;">{chips}</div>', unsafe_allow_html=True)

    per_src = result.get("per_src_conf", [])
    if per_src:
        with st.expander("📊 Confidence breakdown by source"):
            for item in per_src:
                pct   = int(item["score"] * 100)
                color = "#34d399" if item["score"] >= 0.7 else "#fbbf24" if item["score"] >= 0.45 else "#f87171"
                st.markdown(f"""
                <div style="margin:6px 0;">
                    <div style="display:flex;justify-content:space-between;font-size:.8rem;color:#94a3b8;">
                        <span>📄 {item['source']}</span>
                        <span style="color:{color};font-weight:600;">{pct}%</span>
                    </div>
                    <div style="background:#1e293b;border-radius:4px;height:6px;margin-top:3px;">
                        <div style="width:{pct}%;background:{color};height:6px;border-radius:4px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    docs = result.get("docs", [])
    if docs:
        with st.expander("🔬 View retrieved context chunks"):
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source", "?")
                pg  = doc.metadata.get("page", "?")
                sec = doc.metadata.get("section", "")
                st.markdown(f"**Chunk {i}** · `{src}` · Page {pg}" + (f" · *{sec}*" if sec else ""))
                st.code(doc.page_content[:400], language=None)

    # Feedback buttons
    if idx not in st.session_state.feedback_done:
        col1, col2, _ = st.columns([1, 1, 8])
        if col1.button("👍", key=f"up_{idx}"):
            log_feedback(
                query=msg.get("original_query", ""),
                answer=answer, confidence=conf,
                verified=verif.get("verified", False),
                rating="up", sources=sources,
            )
            st.session_state.feedback_done.add(idx)
            st.toast("Thanks! 👍", icon="✅")
            st.rerun()
        if col2.button("👎", key=f"dn_{idx}"):
            log_feedback(
                query=msg.get("original_query", ""),
                answer=answer, confidence=conf,
                verified=verif.get("verified", False),
                rating="down", sources=sources,
            )
            st.session_state.feedback_done.add(idx)
            st.toast("Feedback recorded 👎", icon="📝")
            st.rerun()
    else:
        st.markdown(
            '<div style="color:#475569;font-size:.78rem;margin-top:4px;">✔ Feedback recorded</div>',
            unsafe_allow_html=True,
        )


# ── Render history ────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    render_message(i, msg)

# ── Input ─────────────────────────────────────────────────────────────────────
query = st.chat_input("Ask about HOD, attendance, scholarship, hostel, fees…")

# Also handle quick-prompt sidebar clicks
if "quick_prompt" in st.session_state:
    query = st.session_state.pop("quick_prompt")

if query:
    engine = get_engine()

    # Append user message immediately so it displays
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🧠 Thinking with Advanced RAG Engine…"):
        try:
            result = engine.query(query, metadata_filter=metadata_filter)
            st.session_state.messages.append({
                "role":           "assistant",
                "content":        result["answer"],
                "result":         result,
                "original_query": query,
            })
        except Exception as e:
            st.session_state.messages.append({
                "role":           "assistant",
                "content":        f"⚠️ Error: {e}",
                "result":         {},
                "original_query": query,
            })

    st.rerun()
