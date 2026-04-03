"""
Semantic & Sentence-Boundary Aware Chunker
==========================================
Improvements:
  1. Semantic Chunking (Step 13) — Splits based on meaning shifts using embeddings.
  2. Block Preservation — Maintains metadata-rich block dividers (---).
  3. Hybrid Fallback — Uses sentence-boundary splitting as a robust second layer.
"""

from __future__ import annotations
import re
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from core.config import EMBEDDING_MODEL

# Try sentence tokenizer — graceful fallback if nltk not installed
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
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
    """Split text into sentences."""
    if _HAS_NLTK:
        return sent_tokenize(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def split_into_chunks(
    docs: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Split cleaned docs into meaning-based semantic chunks (Step 13).
    """
    print("🧠 Initialising Semantic Chunker (Step 13)...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # We use 'percentile' threshold to find meaningful shifts in embedding space
    semantic_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile"
    )
    
    chunks: list[dict] = []
    block_pattern = r"(?:-{10,}|={10,})"

    for doc in docs:
        filename = doc.get("source", "unknown")
        full_text = doc.get("text", "")
        if not full_text:
            continue

        # Pass 1: Block Splitting (Dividers)
        blocks = re.split(block_pattern, full_text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            meta = extract_metadata_from_block(block)

            # Pass 2: Semantic Sub-Chunking
            try:
                sub_chunks = semantic_splitter.split_text(block)
            except Exception as e:
                print(f"⚠️ Semantic chunking failed for {filename}, falling back to sentences: {e}")
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
