"""
Text cleaning helpers for ingestion pipeline.
"""

from __future__ import annotations

import re


_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINES = re.compile(r"\n{3,}")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def _normalize_text(text: str) -> str:
    cleaned = _ZERO_WIDTH.sub("", text or "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINES.sub("\n\n", cleaned)
    return cleaned.strip()


def clean_text(docs: list[dict]) -> list[dict]:
    """
    Normalize text while preserving ingestion schema.
    """
    cleaned_docs: list[dict] = []
    for doc in docs:
        text = _normalize_text(doc.get("text", ""))
        if not text:
            continue
        cleaned_docs.append(
            {
                "text": text,
                "source": doc.get("source", "unknown"),
                "page": doc.get("page", 1),
            }
        )
    return cleaned_docs
