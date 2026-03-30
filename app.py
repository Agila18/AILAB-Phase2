"""
🖥️ Agila & Person 1: Streamlit UI
Responsibility: Build an interactive, aesthetically pleasing frontend.

REQUIREMENTS:
- Sidebar with data ingestion trigger (Run ingestion pipeline).
- Main area for Q&A.
- Display answer with distinct styling.
- Expandable section to show citations and text snippets.
- Display verifier results and confidence score clearly.
"""

import streamlit as st
from core.pipeline import run_qa_pipeline, run_ingestion_pipeline

st.set_page_config(page_title="RAG Student Assistant", page_icon="🎓", layout="wide")

st.title("🎓 RAG Student Assistant")
st.markdown("Ask anything based on the official documents. Every answer is verified and cited.")

with st.sidebar:
    st.header("Admin Settings")
    st.markdown("Use this to process new documents in the `data/` folder.")
    if st.button("Re-Index Documents"):
        with st.spinner("Ingesting documents..."):
            run_ingestion_pipeline()
            st.success("Documents re-indexed successfully!")

query = st.text_input("Ask your question here:", placeholder="e.g., What is the scholarship policy?")

if query:
    with st.spinner("Searching and generating answer..."):
        result = run_qa_pipeline(query)
        
        answer = result.get("answer", "")
        context = result.get("context", [])
        verified = result.get("verified", False)
        confidence = result.get("confidence", 0.0)
        
        # Display Verification & Confidence Metrics
        col1, col2 = st.columns(2)
        with col1:
            if verified:
                st.success("✅ **Verified:** Answer grounded in context.")
            else:
                st.error("❌ **Unverified:** Answer may contain hallucinations. Double check sources.")
        with col2:
            st.info(f"📊 **Confidence Score:** {confidence:.2f} / 1.00")

        # Display Answer
        st.subheader("Answer:")
        st.write(answer)
        
        # Display Context & Citations
        with st.expander("📚 View Citations & Sources", expanded=False):
            if context:
                for idx, chunk in enumerate(context):
                    source = chunk.get('source', 'Unknown')
                    page = chunk.get('page', 'N/A')
                    text = chunk.get('text', '')
                    st.markdown(f"**Source {idx+1}:** `{source}` (Page {page})")
                    st.caption(f"> {text}")
                    st.divider()
            else:
                st.write("No relevant context found.")
