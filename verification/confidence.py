"""
Confidence Score — Multi-Signal Scoring Engine (Industry Style)
=============================================================
Factors:
  1. Reranker Score (40%)        → Deep semantic cross-attention
  2. Embedding Similarity (30%) → Dense vector overlap
  3. Keyword Overlap (20%)       → Exact term matching
  4. Context Coverage (10%)     → Breath of source support
"""

from __future__ import annotations
import re
from sklearn.metrics.pairwise import cosine_similarity


def detect_injection(text: str) -> bool:
    """Basic prompt injection protection."""
    patterns = [
        "ignore previous", "forget the above", "system prompt",
        "ignore the context", "new instructions", "disregard the documents",
    ]
    t = text.lower()
    return any(p in t for p in patterns)


def compute_embedding_similarity(answer: str, context: str, embed_model) -> float:
    """Step 1: Calculate cosine similarity in vector space."""
    try:
        a_emb = embed_model.embed_query(answer)
        c_emb = embed_model.embed_query(context)
        return float(cosine_similarity([a_emb], [c_emb])[0][0])
    except:
        return 0.5


def keyword_overlap(answer: str, context: str) -> float:
    """Step 2: Calculate exact word overlap."""
    a_words = set(answer.lower().split())
    c_words = set(context.lower().split())
    return len(a_words & c_words) / max(len(a_words), 1)


def coverage_score(answer: str, context_chunks: list) -> float:
    """Step 3: Percentage of chunks that contain at least one answer keyword."""
    if not context_chunks:
        return 0.0
    count = sum(1 for c in context_chunks if any(w in (c.page_content.lower() if hasattr(c, "page_content") else str(c).lower()) for w in answer.lower().split()))
    return count / len(context_chunks)


def compute_confidence(
    answer: str, 
    docs: list, 
    reranker_score: float, 
    embed_model
) -> float:
    """
    Step 4: Final Industry-Style Multi-Signal Confidence Formula.
    Weighs Reranker, Embeddings, Keywords, and Coverage.
    """
    if not answer or not docs:
        return 0.0

    if detect_injection(answer):
        return 0.0
        
    # Refusal override
    answer_lower = answer.lower()
    refusal_phrases = ["could not find", "not found", "not available", "please contact", "i am a cit student assistant"]
    if any(p in answer_lower for p in refusal_phrases):
        return 0.0

    full_context = " ".join([d.page_content if hasattr(d, "page_content") else str(d) for d in docs])

    # 1. Similarity (30%)
    emb_sim = compute_embedding_similarity(answer, full_context, embed_model)
    
    # 2. Keywords (20%)
    overlap = keyword_overlap(answer, full_context)
    
    # 3. Coverage (10%)
    coverage = coverage_score(answer, docs)

    # 4. Final Combination
    # reranker_score is expected to be normalized 0-1 from rag_engine.py
    confidence = (
        0.4 * reranker_score +
        0.3 * emb_sim +
        0.2 * overlap +
        0.1 * coverage
    )

    return round(confidence, 2)


def compute_per_source_confidence(answer: str, docs, reranker_score: float, embed_model) -> list[dict]:
    """Return per-source breakdown for UI."""
    if not docs:
        return []

    results = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown") if hasattr(doc, "metadata") else "unknown"
        # For single doc, coverage is usually 1 if keywords match
        score = compute_confidence(answer, [doc], reranker_score, embed_model)
        results.append({"source": source, "score": score})

    return sorted(results, key=lambda x: x["score"], reverse=True)
