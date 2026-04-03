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


def compute_confidence(answer: str, context) -> float:
    """
    Compute a 0.0–1.0 confidence score for the generated answer.

    Parameters
    ----------
    answer  : str
    context : list[Document] | list[str] | list[dict]
    """
    if not answer or not context:
        return 0.0

    answer_lower = answer.lower()

    # ── Refusal detection ────────────────────────────────────────────────────
    refusal_phrases = [
        "could not find",
        "not found in the retrieved",
        "not available",
        "please contact the college",
        "i am a cit student assistant",
    ]
    for phrase in refusal_phrases:
        if phrase in answer_lower:
            return 0.0

    # ── Build combined context text ──────────────────────────────────────────
    if isinstance(context[0], str):
        context_text = " ".join(context).lower()
    elif isinstance(context[0], dict):
        context_text = " ".join([c.get("text", "") for c in context]).lower()
    else:
        # LangChain Document objects
        context_text = " ".join([c.page_content for c in context]).lower()

    # ── Factor 1: Keyword Overlap (0.0 → 0.50) ──────────────────────────────
    STOPWORDS = {
        "the", "is", "a", "an", "in", "of", "to", "and", "for", "are",
        "it", "on", "at", "by", "or", "be", "as", "this", "that", "from",
        "with", "was", "has", "have", "can", "will", "not", "but", "if",
        "you", "your", "they", "their", "we", "our", "i", "my", "do",
        "about", "so", "would", "should", "could", "please",
    }
    answer_words = [
        w.strip(".,;:!?\"'()")
        for w in answer_lower.split()
        if len(w) > 2 and w.strip(".,;:!?\"'()") not in STOPWORDS
    ]

    if answer_words:
        matches = sum(1 for w in answer_words if w in context_text)
        overlap = matches / len(answer_words)
    else:
        overlap = 0.0

    keyword_score = min(overlap, 1.0) * 0.50

    # ── Factor 2: Answer Length (0.0 → 0.30) ────────────────────────────────
    word_count = len(answer.split())
    if word_count < 5:
        length_score = 0.05
    elif word_count < 20:
        length_score = 0.15
    elif word_count < 50:
        length_score = 0.25
    else:
        length_score = 0.30

    # ── Factor 3: Source Citation Bonus (0.0 → 0.20) ────────────────────────
    citation_score = 0.0
    citation_markers = ["source", "according to", "as per", ".txt", "document"]
    for marker in citation_markers:
        if marker in answer_lower:
            citation_score = 0.20
            break

    confidence = keyword_score + length_score + citation_score
    return round(min(confidence, 1.0), 2)


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
