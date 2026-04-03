"""
Person 6: Confidence Score
Responsibility: Compute a confidence score (0.0 to 1.0) for the answer.

Uses a combination of:
1. Keyword overlap between answer and context
2. Answer length heuristic (very short = low confidence)
3. Refusal detection (explicit "not found" = 0.0 confidence in content)
"""


def compute_confidence(answer: str, context: list[dict] | list[str]) -> float:
    """
    Computes a confidence score based on the word overlap ratio.
    """
    if not answer or not context:
        return 0.0

    # Build context string
    if isinstance(context[0], str):
        full_context = " ".join(context).lower()
    else:
        full_context = " ".join([c.get("text", "") for c in context]).lower()

    # Calculate Jaccard-like overlap
    answer_words = set(answer.lower().split())
    context_words = set(full_context.split())

    overlap = len(answer_words & context_words)
    total = len(answer_words)

    return overlap / total if total else 0.0

    answer_lower = answer.lower()

    # If the model explicitly refused, confidence in "having an answer" is 0
    # but confidence in the system working correctly is high
    refusal_phrases = [
        "could not find",
        "not found in the retrieved",
        "not available",
        "i am a cit student assistant",
    ]
    for phrase in refusal_phrases:
        if phrase in answer_lower:
            return 0.0  # No answer found — but system behaved correctly

    # Build combined context text
    if isinstance(context[0], dict):
        context_text = " ".join([c.get("text", "") for c in context]).lower()
    else:
        context_text = " ".join([c.page_content for c in context]).lower()

    # --- Factor 1: Keyword Overlap (0.0 to 0.5) ---
    stopwords = {
        "the", "is", "a", "an", "in", "of", "to", "and", "for", "are",
        "it", "on", "at", "by", "or", "be", "as", "this", "that", "from",
        "with", "was", "has", "have", "can", "will", "not", "but", "if",
        "you", "your", "they", "their", "we", "our", "i", "my", "do",
    }
    answer_words = [
        w.strip(".,;:!?\"'()") for w in answer_lower.split()
        if len(w) > 2 and w.strip(".,;:!?\"'()") not in stopwords
    ]

    if answer_words:
        matches = sum(1 for w in answer_words if w in context_text)
        overlap = matches / len(answer_words)
    else:
        overlap = 0.0

    keyword_score = min(overlap, 1.0) * 0.5  # max 0.5 from this factor

    # --- Factor 2: Answer Length (0.0 to 0.3) ---
    word_count = len(answer.split())
    if word_count < 5:
        length_score = 0.05  # Very short answer = low confidence
    elif word_count < 20:
        length_score = 0.15
    elif word_count < 50:
        length_score = 0.25
    else:
        length_score = 0.30  # Detailed answer = higher confidence

    # --- Factor 3: Source Citation Bonus (0.0 to 0.2) ---
    citation_score = 0.0
    citation_markers = ["source", "according to", "as per", ".txt", "document"]
    for marker in citation_markers:
        if marker in answer_lower:
            citation_score = 0.20
            break

    confidence = keyword_score + length_score + citation_score
    return round(min(confidence, 1.0), 2)
