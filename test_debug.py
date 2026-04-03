from rag_engine import RAGEngine
import pprint

engine = RAGEngine()
user_query = "Who is the HOD for the CSE department?"

retrieval_q = "HOD CSE"
best_docs, avg_rerank, avg_sim = engine._retrieve_and_rerank_internal(retrieval_q)
print("Best Docs:")
for d in best_docs:
    print(" -", getattr(d, 'page_content', str(d))[:100].replace('\n', ' '))

context_text = "\n\n---\n\n".join(d.page_content for d in best_docs)
draft_prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nAnswer:"
answer = engine.llm.invoke(draft_prompt)

print("Draft Answer:", answer)

from verification.confidence import compute_confidence
confidence = compute_confidence(answer, best_docs, reranker_score=avg_rerank, embedding_sim=avg_sim)

print("First Pass Confidence:", confidence)
print("Avg Rerank:", avg_rerank)
print("Avg Sim:", avg_sim)

from verification.verifier import verify_answer
v = verify_answer(answer, best_docs)
print("First Pass Verification:", v.get("score"))
