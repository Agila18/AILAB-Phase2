"""
Enhanced Answer Verifier
==========================
Returns a structured result dict instead of a plain bool.

Result schema:
  {
    "verified":              bool,   # overall grounding verdict
    "score":                 float,  # 0.0–1.0 grounding score
    "unsupported_sentences": list,   # sentences NOT found in context
    "supported_sentences":   list,   # sentences found in context
    "support_ratio":         float,  # fraction of sentences grounded
  }
"""

from __future__ import annotations
from verification.span_highlighter import highlight_spans


# Known refusal phrases — system behaved correctly, mark as verified
_REFUSALS = [
    "could not find",
    "not found",
    "please contact",
    "i am a cit student assistant",
    "i could not find this information",
]


def _calculate_relevance(query: str, answer: str) -> float:
    """Heuristic for answer relevance to the query."""
    q_words = set(query.lower().split())
    a_words = set(answer.lower().split())
    if not q_words: return 1.0
    overlap = q_words.intersection(a_words)
    return round(len(overlap) / len(q_words), 2)


def verify_answer(answer: str, docs, query: str = "") -> dict:
    """
    Verify whether the generated answer is grounded in the retrieved documents.

    Returns
    -------
    {
        "verified": bool,
        "score": float, (Faithfulness)
        "relevance": float,
        "unsupported_sentences": list,
        "supported_sentences": list,
    }
    """
    empty_result = {
        "verified": False,
        "score": 0.0,
        "relevance": 0.0,
        "unsupported_sentences": [],
        "supported_sentences": [],
    }

    if not answer or not docs:
        return empty_result

    answer_lower = answer.lower()

    # Explicit refusals → system behaves correctly
    if any(phrase in answer_lower for phrase in _REFUSALS):
        return {
            "verified": True,
            "score": 1.0,
            "relevance": 1.0,
            "unsupported_sentences": [],
            "supported_sentences": [answer],
        }

    # Run span highlighting
    span_result = highlight_spans(answer, docs)
    faithfulness = span_result["support_ratio"]
    relevance = _calculate_relevance(query, answer) if query else 1.0
    
    verified = faithfulness >= 0.5

    # Enrich supported sentences with metadata
    enriched_supported = []
    for item in span_result["supported"]:
        sent_text = item["text"]
        doc_idx = item["doc_idx"]
        
        doc = docs[doc_idx]
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        
        enriched_supported.append({
            "text": sent_text,
            "doc_idx": doc_idx,
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", "?"),
            "section": meta.get("section", "General")
        })

    return {
        "verified": verified,
        "score": round(faithfulness, 2),
        "relevance": relevance,
        "unsupported_sentences": span_result["unsupported"],
        "supported_sentences":   enriched_supported,
    }
