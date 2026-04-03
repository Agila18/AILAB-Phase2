import sys
sys.path.append("D:/aiLAB-PHASE2/rag-assistant")
from rag_engine import RAGEngine

engine = RAGEngine()
query = "Who is the HOD of CSE?"
print(f"Running query: {query}")
try:
    for kind, data in engine.query_with_streaming(query):
        pass
    print("Test passed.")
except Exception as e:
    print(f"Error: {e}")
