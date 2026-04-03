from rag_engine import RAGEngine
import pprint

engine = RAGEngine()
user_query = "Who is the HOD for the CSE department?"
res = engine.query(user_query)

print("\n--- RESULTS ---")
print("Answer:", res.get("answer"))
print("Confidence:", res.get("confidence"))
metrics = res.get("metrics", {})
print("Metrics:", metrics)
verification = res.get("verification", {})
print("Verification score:", verification.get("score"))
