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
    """
    Split answer into sentences and check which document best supports each.
    Returns: {
        "supported": list of {"text": str, "doc_idx": int},
        "unsupported": list of str,
        "support_ratio": float
    }
    """
    if not answer or not docs:
        return {"supported": [], "unsupported": [], "support_ratio": 0.0}

    # Use robust sentence splitter (protects abbreviations like Dr., Prof.)
    sentences = _split_sentences(answer)
    
    if not sentences:
        # Fallback if splitter fails on a single-line answer
        sentences = [answer]

    supported = []
    unsupported = []
    
    doc_contents = []
    for d in docs:
        if hasattr(d, "page_content"): doc_contents.append(d.page_content.lower())
        elif isinstance(d, dict): doc_contents.append(d.get("text", "").lower())
        else: doc_contents.append(str(d).lower())

    for sent in sentences:
        sent_lower = sent.lower()
        best_doc_idx = -1
        max_overlap = 0.0
        
        words = _meaningful_words(sent)
        if len(words) < _MIN_WORDS:
            # For very short sentences, require exact match or just skip
            best_doc_idx = -1
        else:
            for idx, doc_text in enumerate(doc_contents):
                # 1. Exact match (strongest)
                if sent_lower in doc_text:
                    best_doc_idx = idx
                    max_overlap = 1.0
                    break
                
                # 2. Token overlap (Jaccard-like)
                matches = sum(1 for w in words if w in doc_text)
                overlap = matches / len(words)
                
                if overlap > max_overlap and overlap >= _SUPPORT_THRESHOLD:
                    max_overlap = overlap
                    best_doc_idx = idx
        
        if best_doc_idx != -1:
            supported.append({"text": sent, "doc_idx": best_doc_idx})
        else:
            unsupported.append(sent)

    support_ratio = len(supported) / len(sentences)
    
    return {
        "supported": supported,
        "unsupported": unsupported,
        "support_ratio": round(support_ratio, 2)
    }
