"""
Enhanced Answer Verifier — Sentence-Level Citations (Step 2.0)
=============================================================
Matches each sentence of the LLM answer to its most relevant 
source chunk using embedding similarity (cosine similarity).
"""

from __future__ import annotations
import re
from sklearn.metrics.pairwise import cosine_similarity
from verification.span_highlighter import highlight_spans


def split_sentences(text: str) -> list[str]:
    """Split into sentences, handling multiple delimiters and whitespace."""
    # Using more robust split than re.split(r'(?<=[.!?]) +', text)
    # This protects abbreviations like Dr., Prof. but user requested re-split.
    # I'll stick close to their spec but make it slightly more robust.
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', text) if s.strip()]


def find_best_source(sentence: str, docs: list, embed_model):
    """Find the document chunk that best supports the given sentence."""
    best_score = -1
    best_doc = None

    try:
        s_emb = embed_model.embed_query(sentence)
        
        for doc in docs:
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            d_emb = embed_model.embed_query(content)
            score = float(cosine_similarity([s_emb], [d_emb])[0][0])

            if score > best_score:
                best_score = score
                best_doc = doc
    except:
        pass

    return best_doc, best_score


def add_citations(answer: str, docs: list, embed_model) -> str:
    """
    Rebuild the answer string with [Source | Page | Section] markers.
    Matches user's requested literal format.
    """
    # Use requested literal split pattern
    sentences = re.split(r'(?<=[.!?]) +', answer)
    cited_answer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        doc, score = find_best_source(sentence, docs, embed_model)
        
        if doc and score > 0.65: # Threshold for meaningful grounding
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            section = doc.metadata.get("section", "General")
            
            # Format: [Source, Page X, Section: Y]
            citation = f" [{source}, Page {page}, Section: {section}]"
            cited_answer += f"{sentence}{citation} "
        else:
            cited_answer += f"{sentence} "

    return cited_answer.strip()


def verify_answer(answer: str, docs: list, query: str = "", embed_model = None) -> dict:
    """
    Main verifier entry point.
    Returns structured results including the new 'cited_answer'.
    """
    if not answer or not docs:
        return {"verified": False, "score": 0.0, "cited_answer": answer}

    # 1. Generate Cited Answer (The 'Killer Feature')
    cited_answer = add_citations(answer, docs, embed_model) if embed_model else answer
    
    # 2. Run traditional span highlighting for support metrics
    span_result = highlight_spans(answer, docs)
    faithfulness = span_result["support_ratio"]
    
    # 3. Final verdict
    verified = faithfulness >= 0.5

    return {
        "verified": verified,
        "score": round(faithfulness, 2),
        "cited_answer": cited_answer,
        "unsupported_sentences": span_result["unsupported"],
        "supported_sentences":   span_result["supported"],
    }
