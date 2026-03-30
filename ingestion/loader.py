"""
👨‍💻 Person 1: Document Loader
Responsibility: Load text from PDF, DOCX, TXT files.

CRITICAL REQUIREMENT:
Output MUST be a list of dictionaries with this exact format:
[
  {
    "text": "Extracted text content...",
    "source": "filename.pdf",
    "page": 1  # or section name, or None if TXT
  }
]

INNOVATION OPPORTUNITIES:
- Identify tables or images (OCR) in PDFs.
- Use advanced loaders like unstructured.io.
- Add metadata like document creation date.
- Handle exceptions and malformed files gracefully.
"""

import os
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def load_documents(data_dir: str) -> list[dict]:
    """
    Loads documents from the given directory and returns them in the standard format.
    
    Args:
        data_dir (str): Path to the directory containing input files.
        
    Returns:
        list[dict]: List of document dictionaries containing text, source, and page.
    """
    docs = []
    
    # TODO: Implement loading logic here.
    # 1. Iterate over files in data_dir
    # 2. Use the appropriate loader based on the file extension (.pdf, .docx, .txt)
    # 3. Transform the loader's output into the uniquely required dictionary format.
    
    # Example snippet:
    # docs.append({"text": "dummy text", "source": "test.pdf", "page": 1})
    
    return docs
