"""
Document loading utilities for ingestion.

Output contract:
[
  {
    "text": "...",
    "source": "filename.ext",
    "page": 1
  }
]
"""

from __future__ import annotations

import os
from html.parser import HTMLParser
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".html", ".htm"}


class _SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self.parts.append(data.strip())

    def get_text(self) -> str:
        return "\n".join(self.parts)


def _load_one_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix in {".html", ".htm"}:
        html_text = path.read_text(encoding="utf-8", errors="ignore")
        parser = _SimpleHTMLTextExtractor()
        parser.feed(html_text)
        return [
            {
                "text": parser.get_text(),
                "source": path.name,
                "page": 1,
            }
        ]
    else:
        return []

    results: list[dict] = []
    for idx, doc in enumerate(loader.load()):
        source_name = os.path.basename(doc.metadata.get("source", str(path)))
        # PDF loaders usually provide page in metadata; other formats may not.
        page = int(doc.metadata.get("page", idx + 1))
        results.append(
            {
                "text": doc.page_content or "",
                "source": source_name,
                "page": page,
            }
        )
    return results


def load_documents(data_dir: str) -> list[dict]:
    """
    Load all supported documents from the input directory.
    """
    directory = Path(data_dir)
    if not directory.exists() or not directory.is_dir():
        return []

    documents: list[dict] = []
    for path in sorted(directory.glob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            documents.extend(_load_one_file(path))
        except Exception as exc:
            # Continue loading other files even when one is malformed.
            print(f"[loader] Skipping {path.name}: {exc}")
    return documents
