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


def verify_answer(answer: str, docs) -> dict:
    """
    Verify whether the generated answer is grounded in the retrieved documents.

    Parameters
    ----------
    answer : str
    docs   : list[Document] | list[str]

    Returns
    -------
    dict — see module docstring for schema
    """
    empty_result = {
        "verified": False,
        "score": 0.0,
        "unsupported_sentences": [],
        "supported_sentences": [],
        "support_ratio": 0.0,
    }

    if not answer or not docs:
        return empty_result

    answer_lower = answer.lower()

    # Explicit refusals → system is working correctly, mark as verified
    if any(phrase in answer_lower for phrase in _REFUSALS):
        return {
            "verified": True,
            "score": 1.0,
            "unsupported_sentences": [],
            "supported_sentences": [answer],
            "support_ratio": 1.0,
        }

    # Run span highlighting
    span_result = highlight_spans(answer, docs)

    support_ratio = span_result["support_ratio"]
    verified = support_ratio >= 0.5  # At least half the sentences must be grounded

    return {
        "verified": verified,
        "score": round(support_ratio, 2),
        "unsupported_sentences": span_result["unsupported"],
        "supported_sentences":   span_result["supported"],
        "support_ratio":         support_ratio,
    }
