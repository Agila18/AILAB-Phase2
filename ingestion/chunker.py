"""
Chunking module for ingestion pipeline.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_into_chunks(
    docs: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Split cleaned docs into overlapping chunks while preserving metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for doc in docs:
        text = doc.get("text", "")
        if not text:
            continue
        parts = splitter.split_text(text)
        for part in parts:
            normalized = part.strip()
            if not normalized:
                continue
            chunks.append(
                {
                    "text": normalized,
                    "source": doc.get("source", "unknown"),
                    "page": doc.get("page", 1),
                }
            )
    return chunks
