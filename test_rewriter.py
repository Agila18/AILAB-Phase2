import sys
from rag_engine import RAGEngine
from query.rewriter import rewrite_query

engine = RAGEngine()
user_query = "Who is the HOD for the CSE department?"
q, i, label = rewrite_query(user_query, llm=engine.llm)
print("Rewritten query:", q)
print("Intent:", i)
