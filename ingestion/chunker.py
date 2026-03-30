"""
👨‍💻 Person 2: Text Chunker (Splitting)
Responsibility: Split cleaned text into smaller, manageable chunks.

CRITICAL REQUIREMENT:
Takes the cleaned documents and splits them into chunks.
Output MUST maintain the standard format:
[
  {
    "text": "Chunk text...",
    "source": "filename.pdf",
    "page": 1
  }
]

INNOVATION OPPORTUNITIES:
- Implement semantic chunking instead of just fixed-size.
- Ensure sentence boundaries are not broken.
- Experiment with overlapping boundaries to keep context.
"""

# from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_into_chunks(docs: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    """
    Splits document text into manageable chunks.
    
    Args:
        docs (list[dict]): Cleaned documents.
        chunk_size (int): Max size of a chunk.
        chunk_overlap (int): Overlap between chunks.
        
    Returns:
        list[dict]: Chunked documents.
    """
    chunks = []
    
    # TODO: Implement chunking logic here 
    # 1. Initialize your text splitter
    # 2. Split doc['text'] 
    # 3. Re-pack the split text into the required dictionary format retaining source and page.
    
    return chunks
