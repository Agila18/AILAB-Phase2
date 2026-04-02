"""
Hybrid Retriever: Combines semantic (vector) search with BM25 keyword search
for more accurate document retrieval.
"""

from rank_bm25 import BM25Okapi


def build_bm25(docs):
    """Build a BM25 index from a list of LangChain Document objects."""
    texts = [doc.page_content.split() for doc in docs]
    return BM25Okapi(texts), docs


def hybrid_search(query, vectorstore, bm25, bm25_docs, k=8):
    """
    Perform hybrid search combining semantic similarity and BM25 keyword matching.
    Returns deduplicated results from both methods.
    """
    # Semantic search (vector similarity)
    semantic_docs = vectorstore.similarity_search(query, k=k)

    # BM25 keyword search
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    keyword_docs = [bm25_docs[i] for i in top_n]

    # Merge and deduplicate
    combined = semantic_docs + keyword_docs
    seen = set()
    final_docs = []
    for doc in combined:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            final_docs.append(doc)

    return final_docs[:k]
