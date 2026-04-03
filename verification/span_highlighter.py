"""
Answer Span Highlighter — Fixed sentence splitting (handles Dr., Mr., etc.)
"""
from __future__ import annotations
import re

_STOPWORDS = {
    "the","is","a","an","in","of","to","and","for","are","it","on","at","by",
    "or","be","as","this","that","from","with","was","has","have","can","will",
    "not","but","if","you","your","they","their","we","our","i","my","do",
    "about","so","would","should","could","please","also",
}
_MIN_WORDS = 3
_SUPPORT_THRESHOLD = 0.30   # lowered from 0.35 — more forgiving


def _split_sentences(text: str) -> list[str]:
    """Split into sentences, skipping common abbreviations (Dr., Mr., etc.)."""
    # Protect common abbreviations before splitting
    protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|St|Dept|No|vs|etc)\.',
                       r'\1<DOT>', text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    # Restore
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]


def _meaningful_words(sentence: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", sentence.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


def highlight_spans(answer: str, docs) -> dict:
    if not answer or not docs:
        return {"supported": [], "unsupported": [], "support_ratio": 0.0}

    if isinstance(docs[0], str):
        context = " ".join(docs).lower()
    else:
        context = " ".join(d.page_content for d in docs).lower()

    sentences = _split_sentences(answer)
    supported: list[str] = []
    unsupported: list[str] = []

    for sent in sentences:
        words = _meaningful_words(sent)
        if len(words) < _MIN_WORDS:
            supported.append(sent)   # trivially short → don't penalise
            continue
        matches = sum(1 for w in words if w in context)
        if matches / len(words) >= _SUPPORT_THRESHOLD:
            supported.append(sent)
        else:
            unsupported.append(sent)

    total = len(sentences)
    ratio = round(len(supported) / total, 2) if total else 0.0
    return {"supported": supported, "unsupported": unsupported, "support_ratio": ratio}
