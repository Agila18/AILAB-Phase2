"""
Multi-Hop Retrieval
=====================
When confidence is low on the first answer, extract key entities/terms
from the draft answer and perform a second retrieval pass.

This helps when a question spans multiple documents or requires chaining
facts from different sources.

Flow:
  Question
    → First retrieve+rerank → Draft answer (confidence < threshold)
    → Extract key entities from draft
    → Second retrieve+rerank with entity-enriched query
    → Merge unique docs from both passes
    → Return combined top-k
"""

from __future__ import annotations
import re

# Confidence threshold below which multi-hop kicks in
MULTI_HOP_THRESHOLD = 0.45

# Stopwords to skip when extracting entities from draft
_STOPWORDS = {
    "the", "is", "a", "an", "in", "of", "to", "and", "for", "are",
    "it", "on", "at", "by", "or", "be", "as", "this", "that", "from",
    "with", "was", "has", "have", "can", "will", "not", "but", "if",
    "you", "your", "they", "their", "we", "our", "i", "my", "do",
    "about", "so", "would", "should", "could", "please", "also",
    "college", "cit", "student", "students",
}


def extract_entities(text: str, max_terms: int = 8) -> list[str]:
    """
    Extract meaningful noun-like terms from a text snippet.
    Simple heuristic: words > 4 chars, not stopwords, preferring capitalised.
    """
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text)
    seen: set[str] = set()
    entities: list[str] = []

    # Prefer capitalised (likely proper nouns / domain terms)
    for tok in tokens:
        lower = tok.lower()
        if lower in _STOPWORDS or lower in seen or len(tok) < 4:
            continue
        seen.add(lower)
        if tok[0].isupper():
            entities.insert(0, lower)
        else:
            entities.append(lower)

        if len(entities) >= max_terms:
            break

    return entities


def multi_hop_retrieve(
    original_query: str,
    draft_answer: str,
    confidence: float,
    retrieve_fn,          # callable: (query: str) -> list[Document]
    existing_docs: list,
) -> list:
    """
    Optionally perform a second retrieval pass if confidence is below threshold.

    Parameters
    ----------
    original_query : str
    draft_answer   : str    — first-pass answer from LLM
    confidence     : float  — confidence of the draft answer
    retrieve_fn    : callable that takes a query string and returns list[Document]
    existing_docs  : list[Document] — docs from the first retrieval pass

    Returns
    -------
    list[Document] — merged, deduplicated docs (first pass + second pass)
    """
    if confidence >= MULTI_HOP_THRESHOLD:
        return existing_docs  # No hop needed

    print(f"🔄 Multi-hop triggered (conf={confidence:.2f}). Extracting entities…")

    entities = extract_entities(draft_answer)
    if not entities:
        print("⚠️  No entities found for multi-hop. Keeping first-pass docs.")
        return existing_docs

    hop_query = original_query + " " + " ".join(entities)
    print(f"   Hop query: '{hop_query[:80]}…'")

    try:
        hop_docs = retrieve_fn(hop_query)
    except Exception as e:
        print(f"⚠️  Multi-hop retrieval failed: {e}")
        return existing_docs

    # Merge and deduplicate
    seen_keys: set[str] = {d.page_content[:120] for d in existing_docs}
    merged = list(existing_docs)

    for doc in hop_docs:
        key = doc.page_content[:120]
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(doc)

    print(f"✅ Multi-hop added {len(merged) - len(existing_docs)} new doc(s). Total: {len(merged)}")
    return merged
