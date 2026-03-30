"""
👨‍💻 Person 5: Retriever (🚨 CORE SYSTEM COMPONENT 🚨)
Responsibility: Retrieve the top-k most relevant chunks for a user query.

⚠️ WHY THIS IS CRITICAL (THE MOST IMPORTANT ROLE):
1. If retrieval fails -> the whole system fails ❌
2. If the retriever fetches the wrong info -> the generator has nothing to work with ❌

CRITICAL REQUIREMENT:
Input: User query, Chroma DB.
Output: A list of relevant chunks in the STANDARD FORMAT:
[
  {
    "text": "...",
    "source": "file.pdf",
    "page": 1
  }
]

INNOVATION OPPORTUNITIES:
- Implement MMR (Max Marginal Relevance) instead of basic similarity search for diversity.
- Expose search score to the team to help the verifier.
"""

def retrieve_docs(query: str, db, top_k: int = 4) -> list[dict]:
    """
    Retrieves the most relevant chunks for the query.
    
    Args:
        query (str): The user's question.
        db: The loaded vector database.
        top_k (int): Number of chunks to retrieve.
        
    Returns:
        list[dict]: Standard format chunks.
    """
    context = []
    
    # TODO: Perform similarity search on the DB using the query.
    # results = db.similarity_search(query, k=top_k)
    
    # TODO: Extract text, source, and page from the results and build the list of dicts.
    # context.append({"text": res.page_content, "source": res.metadata["source"], "page": res.metadata["page"]})
    
    return context
