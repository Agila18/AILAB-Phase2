"""
Sentence-Boundary Aware Chunker
=================================
Improvements over the previous version:
  1. Sentence-boundary splitting — chunks never cut mid-sentence
  2. Section-block splitting (dividers) as the outer layer (preserved)
  3. Metadata extraction from blocks (preserved)
  4. Graceful fallback if nltk.sent_tokenize is unavailable
"""

from __future__ import annotations
import re

# Try sentence tokenizer — graceful fallback if nltk not installed
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False


def extract_metadata_from_block(text: str) -> dict:
    """Extract SECTION, Department, and Keywords from a block of text."""
    metadata = {}
    section_match = re.search(r"SECTION:\s*(.*)", text, re.IGNORECASE)
    if section_match:
        metadata["section"] = section_match.group(1).strip()
    dept_match = re.search(r"Department:\s*(.*)", text, re.IGNORECASE)
    if dept_match:
        metadata["department"] = dept_match.group(1).strip()
    keywords_match = re.search(r"Keywords:\s*(.*)", text, re.IGNORECASE)
    if keywords_match:
        metadata["keywords"] = keywords_match.group(1).strip()
    return metadata


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, with fallback to period-split."""
    if _HAS_NLTK:
        return sent_tokenize(text)
    # Fallback: split on ". " while preserving abbreviation edge cases
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _sentence_aware_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Build chunks that always end on a sentence boundary.

    Algorithm:
      - Tokenize into sentences
      - Greedily fill chunks up to chunk_size characters
      - When a chunk is full, start a new one (with overlap = last N sentences)
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        # If adding this sentence would exceed the limit AND we have content, flush
        if current_len + sent_len > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Overlap: keep trailing sentences whose total length ≤ chunk_overlap
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_sentences = overlap_sentences
            current_len = overlap_len

        current_sentences.append(sent)
        current_len += sent_len

    # Flush remainder
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def split_into_chunks(
    docs: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Split cleaned docs into overlapping, sentence-boundary-aware chunks.

    Pipeline per document:
      1. Split into semantic blocks by divider lines (--- or ===)
      2. Extract metadata from each block
      3. Split each block into sentence-boundary-aware chunks
      4. Prepend metadata header to each chunk text
    """
    chunks: list[dict] = []
    block_pattern = r"(?:-{10,}|={10,})"

    for doc in docs:
        filename = doc.get("source", "unknown")
        full_text = doc.get("text", "")
        if not full_text:
            continue

        blocks = re.split(block_pattern, full_text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            meta = extract_metadata_from_block(block)

            # Sentence-boundary aware sub-chunking
            sub_chunks = _sentence_aware_chunks(block, chunk_size, chunk_overlap)

            # Fallback: if sentence chunker returns nothing, keep the block as-is
            if not sub_chunks:
                sub_chunks = [block]

            for sub_chunk in sub_chunks:
                sub_chunk = sub_chunk.strip()
                if not sub_chunk:
                    continue

                # Prepend metadata header
                header = ""
                if "section" in meta:
                    header += f"[SECTION: {meta['section']}] "
                if "department" in meta:
                    header += f"[Department: {meta['department']}] "
                if "keywords" in meta:
                    header += f"[Keywords: {meta['keywords']}] "

                enriched_text = f"{header}\n{sub_chunk}" if header else sub_chunk

                chunks.append({
                    "text": enriched_text,
                    "source": filename,
                    "page": doc.get("page", 1),
                    **meta,
                })

    return chunks
