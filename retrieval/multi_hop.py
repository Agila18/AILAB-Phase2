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


def group_chunks_by_topic(chunks: list, llm) -> dict[str, list]:
    """
    Step 2: Group by topic.
    Groups retrieved chunks into logical thematic topics using the LLM.
    Returns: { "Topic A": ["Text 1", "Text 2"], "Topic B": ["Text 3"] }
    """
    if not chunks or not llm:
        return {"General": [c.page_content if hasattr(c, "page_content") else str(c) for c in chunks]}

    context_summary = "\n".join([f"- {c.page_content[:200]}..." for c in chunks])
    prompt = (
        "Identify 2-3 specific key topics covered in these document snippets. "
        "Return ONLY a comma-separated list of topic names.\n\n"
        f"Snippets:\n{context_summary}\n\nTopics:"
    )
    
    try:
        topic_names = [t.strip() for t in llm.invoke(prompt).split(",")]
        groups = {t: [] for t in topic_names}
        groups["General"] = []
        
        for c in chunks:
            content = c.page_content if hasattr(c, "page_content") else str(c)
            assigned = False
            for t in topic_names:
                if t.lower() in content.lower():
                    groups[t].append(content)
                    assigned = True
                    break
            if not assigned:
                groups["General"].append(content)
        
        # Clean up empty groups
        return {k: v for k, v in groups.items() if v}
    except:
        return {"General": [c.page_content if hasattr(c, "page_content") else str(c) for c in chunks]}


def process_multi_hop(query: str, chunks: list, llm) -> str:
    """
    Step 3: Combine answers logically.
    Takes the grouped topics and asks the LLM to synthesize a logical, multi-source answer.
    """
    groups = group_chunks_by_topic(chunks, llm)
    
    # Build structured reasoning prompt
    synthesis_blocks = []
    for topic, texts in groups.items():
        block = f"### Topic: {topic}\n" + "\n".join(texts)
        synthesis_blocks.append(block)
    
    context_text = "\n\n".join(synthesis_blocks)
    prompt = (
        "You are an expert CIT Student Assistant. Your task is to synthesize a single, cohesive answer "
        "by reasoning across the following thematic topics extracted from diverse documents.\n\n"
        "GUIDELINES:\n"
        "1. Combine information from multiple topics into a logical narrative.\n"
        "2. If topics conflict, prioritize more recent or specific rules.\n"
        "3. Maintain a helpful, academic, and professional tone.\n\n"
        f"USER QUESTION: {query}\n\n"
        f"GROUPED EVIDENCE:\n{context_text}\n\n"
        "COMPREHENSIVE ANSWER:"
    )
    
    return llm.invoke(prompt)


def multi_hop_retrieve(
    original_query: str,
    draft_answer: str,
    confidence: float,
    retrieve_fn,
    existing_docs: list,
) -> list:
    """
    Step 1: Retrieve chunks (Second pass).
    Main entry point for multi-hop retrieval pass if confidence is low.
    """
    if confidence >= MULTI_HOP_THRESHOLD:
        return existing_docs

    print(f"🔄 Multi-hop (Group & Combine) triggered.")
    entities = extract_entities(draft_answer)
    if not entities:
        return existing_docs

    hop_query = original_query + " " + " ".join(entities)
    try:
        hop_docs = retrieve_fn(hop_query)
    except:
        return existing_docs

    # Merge and deduplicate
    seen_keys: set[str] = {d.page_content[:120] for d in existing_docs}
    merged = list(existing_docs)
    for doc in hop_docs:
        if doc.page_content[:120] not in seen_keys:
            merged.append(doc)
    return merged
