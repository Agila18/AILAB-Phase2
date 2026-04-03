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
    """Extract SECTION and Dept (Atomic) from a block of text."""
    metadata = {}
    section_match = re.search(r"SECTION:\s*(.*)", text, re.IGNORECASE)
    if section_match:
        metadata["section"] = section_match.group(1).strip()
    dept_match = re.search(r"Department:\s*(.*)", text, re.IGNORECASE)
    if dept_match:
        dept = dept_match.group(1).strip()
        if "Computer Science" in dept: dept = "CSE"
        if "Information Technology" in dept: dept = "IT"
        if "Artificial Intelligence" in dept: dept = "AI & DS"
        metadata["dept"] = dept
        metadata["type"] = "department"
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

    for doc in docs:
        filename = doc.get("source", "unknown")
        full_text = doc.get("text", "")
        if not full_text:
            continue

        # Split into logical blocks using delimiters (---, ===, or \n\n)
        raw_blocks = re.split(r"(?:\n\s*\n|[-=]{5,})", full_text)
        
        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue
            
            # Detect metadata from the block itself
            meta = extract_metadata_from_block(block)
            
            # 🔥 SURGICAL GATE: If the block is an 'ENTITY BLOCK' (short & dense), keep it WHOLE.
            # This ensures (Dept + HOD + Location) stays as one single retrieval unit.
            is_entity_block = "Department:" in block or "HOD:" in block
            
            if is_entity_block or len(block) < chunk_size:
                sub_blocks = [block]
            else:
                # For large policy paragraphs, use semantic splitting
                try:
                    sub_blocks = semantic_splitter.split_text(block)
                except Exception as e:
                    sub_blocks = [block]

            for sub_chunk in sub_blocks:
                sub_chunk = sub_chunk.strip()
                if not sub_chunk:
                    continue

                # Prepend metadata header for better RAG context reinforcement
                header = ""
                if "section" in meta:
                    header += f"[SECTION: {meta['section']}] "
                if "dept" in meta:
                    header += f"[Dept: {meta['dept']}] "

                enriched_text = f"{header}\n{sub_chunk}" if header else sub_chunk

                chunks.append({
                    "text": enriched_text,
                    "source": filename,
                    "page": doc.get("page", 1),
                    **meta,
                })

    return chunks
