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


def find_best_source_from_emb(s_emb, docs: list, precalculated_embeddings: list = None):
    """Find the document chunk that best supports the given sentence embedding."""
    best_score = -1
    best_doc = None

    try:
        for idx, doc in enumerate(docs):
            if precalculated_embeddings and idx < len(precalculated_embeddings):
                d_emb = precalculated_embeddings[idx]
                score = float(cosine_similarity([s_emb], [d_emb])[0][0])
                if score > best_score:
                    best_score = score
                    best_doc = doc
    except Exception as e:
        print(f"Verifier Error: {e}")
        pass

    return best_doc, best_score


def add_citations(answer: str, docs: list, embed_model, precalculated_embeddings: list = None) -> str:
    """
    Rebuild the answer string with [Source | Page | Section] markers using batch embeddings.
    """
    # 1. Split sentences
    raw_sentences = re.split(r'(?<=[.!?]) +', answer)
    sentence_data = [] # List of (text, index_in_stream)
    
    # 2. Filter and prepare for batching
    for s in raw_sentences:
        s_strip = s.strip()
        if s_strip and len(s_strip) >= 10:
            sentence_data.append({"text": s_strip, "needs_emb": True})
        else:
            sentence_data.append({"text": s, "needs_emb": False})

    # 3. Batch Embed
    texts_to_embed = [s["text"] for s in sentence_data if s["needs_emb"]]
    if texts_to_embed and embed_model:
        try:
            embs = embed_model.embed_documents(texts_to_embed)
            emb_iter = iter(embs)
            for s in sentence_data:
                if s["needs_emb"]:
                    s["emb"] = next(emb_iter)
        except Exception as e:
            print(f"Batch Embedding Error: {e}")
            for s in sentence_data: s["needs_emb"] = False # Fallback to no citations

    # 4. Reconstruct with citations
    cited_answer = ""
    for s in sentence_data:
        if not s["needs_emb"]:
            cited_answer += f"{s['text']} "
            continue
            
        doc, score = find_best_source_from_emb(s["emb"], docs, precalculated_embeddings)
        
        if doc and score > 0.65: 
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            section = doc.metadata.get("section", "General")
            citation = f" [{source}, Page {page}, Section: {section}]"
            cited_answer += f"{s['text']}{citation} "
        else:
            cited_answer += f"{s['text']} "

    return cited_answer.strip()


def verify_answer(answer: str, docs: list, query: str = "", embed_model = None, precalculated_embeddings: list = None) -> dict:
    """
    Main verifier entry point.
    Returns structured results including the new 'cited_answer'.
    """
    if not answer or not docs:
        return {"verified": False, "score": 0.0, "cited_answer": answer}

    # 1. Generate Cited Answer (The 'Killer Feature')
    cited_answer = add_citations(answer, docs, embed_model, precalculated_embeddings) if embed_model else answer
    
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
