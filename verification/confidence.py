"""
Confidence Score — Multi-Factor Engine
========================================
Factors:
  1. Keyword overlap (answer words found in context)   → up to 0.50
  2. Answer length heuristic                           → up to 0.30
  3. Source citation bonus                             → up to 0.20
  4. Refusal detection (explicit "not found")          →  0.0 override
"""

from __future__ import annotations


def detect_injection(text: str) -> bool:
    """Basic prompt injection protection."""
    patterns = [
        "ignore previous",
        "forget the above",
        "system prompt",
        "ignore the context",
        "new instructions",
        "disregard the documents",
    ]
    t = text.lower()
    return any(p in t for p in patterns)


def compute_confidence(
    answer: str,
    context,
    reranker_score: float | None = None,
    embedding_sim: float | None = None
) -> float:
    """
    Compute an Industry-Level hybrid confidence score (0.0 to 1.0).
    Formula: 0.4*Reranker + 0.3*Similarity + 0.3*Keywords
    """
    if not answer or not context:
        return 0.0

    if detect_injection(answer):
        return 0.0

    answer_lower = answer.lower()

    # ── Refusal detection ────────────────────────────────────────────────────
    refusal_phrases = ["could not find", "not found", "not available", "please contact", "i am a cit student assistant"]
    if any(p in answer_lower for p in refusal_phrases):
        return 0.0

    # ── Factor 1: Keyword Overlap (30% weight) ──────────────────────────────
    texts = []
    for c in context:
        if isinstance(c, str): texts.append(c)
        elif hasattr(c, "page_content"): texts.append(c.page_content)
        elif isinstance(c, dict): texts.append(c.get("text", ""))
        else: texts.append(str(c))
    
    context_text = " ".join(texts).lower()
    STOPWORDS = {"the","is","a","an","in","of","to","and","for","are","it","on","at","by","or","be","as","this","that","from"}
    
    a_words = [w.strip(".,;") for w in answer_lower.split() if w not in STOPWORDS and len(w) > 2]
    if a_words:
        matches = sum(1 for w in a_words if w in context_text)
        keyword_score = matches / len(a_words)
    else:
        keyword_score = 0.0

    # ── Factor 2: Reranker Priority (40% weight) ─────────────────────────────
    # ms-marco reranker scores usually sit between 0-0.9+ for high relevance
    r_score = reranker_score if reranker_score is not None else 0.5

    # ── Factor 3: Embedding Similarity (30% weight) ──────────────────────────
    # Cosine similarity for nomic/ollama is usually 0.5 to 0.8
    e_score = embedding_sim if embedding_sim is not None else 0.5

    # ── Final Industry-Level Hybrid Weighting ────────────────────────────────
    confidence = (0.4 * r_score) + (0.3 * e_score) + (0.3 * keyword_score)
    return round(min(max(confidence, 0.0), 1.0), 2)


def compute_per_source_confidence(answer: str, docs) -> list[dict]:
    """
    Return per-source confidence breakdown for UI display.

    Returns list of {source, score} dicts sorted by score descending.
    """
    if not docs:
        return []

    results = []
    for doc in docs:
        source = (
            doc.metadata.get("source", "unknown")
            if hasattr(doc, "metadata")
            else str(doc)[:40]
        )
        score = compute_confidence(answer, [doc])
        results.append({"source": source, "score": score})

    return sorted(results, key=lambda x: x["score"], reverse=True)
