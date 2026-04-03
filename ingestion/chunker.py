"""
Chunking module for ingestion pipeline with metadata extraction.
"""

from __future__ import annotations
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_metadata_from_block(text: str) -> dict:
    """Extract SECTION, Department, and Keywords from a block of text."""
    metadata = {}
    
    # Extract Section
    section_match = re.search(r"SECTION:\s*(.*)", text, re.IGNORECASE)
    if section_match:
        metadata["section"] = section_match.group(1).strip()
        
    # Extract Department
    dept_match = re.search(r"Department:\s*(.*)", text, re.IGNORECASE)
    if dept_match:
        metadata["department"] = dept_match.group(1).strip()
        
    # Extract Keywords
    keywords_match = re.search(r"Keywords:\s*(.*)", text, re.IGNORECASE)
    if keywords_match:
        metadata["keywords"] = keywords_match.group(1).strip()
        
    return metadata


def split_into_chunks(
    docs: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Split cleaned docs into blocks first (by dividers), then into overlapping chunks,
    extracting and prepending metadata for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    # Dividers used in the text files
    block_pattern = r"(?:-{10,}|={10,})"

    for doc in docs:
        filename = doc.get("source", "unknown")
        full_text = doc.get("text", "")
        if not full_text:
            continue
            
        # Split into semantic blocks first
        blocks = re.split(block_pattern, full_text)
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
                
            # Extract metadata common to this block
            meta = extract_metadata_from_block(block)
            
            # Further split into chunks if block is too large
            sub_chunks = splitter.split_text(block)
            
            for sub_chunk in sub_chunks:
                sub_chunk = sub_chunk.strip()
                if not sub_chunk:
                    continue
                    
                # PREPEND metadata to text for "extra-strength" semantic representation
                header = ""
                if "section" in meta:
                    header += f"[SECTION: {meta['section']}] "
                if "department" in meta:
                    header += f"[Department: {meta['department']}] "
                if "keywords" in meta:
                    header += f"[Keywords: {meta['keywords']}] "
                
                enriched_text = f"{header}\n{sub_chunk}"
                
                chunks.append(
                    {
                        "text": enriched_text,
                        "source": filename,
                        "page": doc.get("page", 1),
                        **meta  # Also store as individual metadata fields
                    }
                )
    return chunks
