"""
🚀 Unified RAG Engine — Advanced v3
======================================
This engine coordinates retrieval, reranking, and verification with high-fidelity citations.
"""

from __future__ import annotations
import os
import pickle
import time
from typing import Iterator

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

from core.config import DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, MODEL_NAME
from query.rewriter import rewrite_query
from verification.confidence import compute_confidence, compute_per_source_confidence
from verification.verifier import verify_answer
from retrieval.multi_hop import multi_hop_retrieve

BM25_CACHE  = "bm25_docs.pkl"
TOP_K       = 5
CANDIDATE_K = 20

# ── Cross-Encoder — loaded once at module level ───────────────────────────────
print("🧠 Loading Cross-Encoder reranker…")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("✅ Reranker ready.")


def section_filter_hint(query: str) -> str | None:
    """Return a source-file hint when the intent is unambiguous."""
    q = query.lower()
    if any(w in q for w in ["hod", "head of department", "faculty"]):
        return "CIT_Academic_Calendar.txt"
    if "attendance" in q:
        return "attendance_rules.txt"
    if "scholarship" in q:
        return "scholarship.txt"
    if "fee" in q or "fees" in q:
        return "fee_structure.txt"
    return None


class RAGEngine:
    def __init__(self):
        self.embeddings  = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
        )
        self.llm = OllamaLLM(model=MODEL_NAME, temperature=0)
        self._load_bm25()

    def _load_bm25(self):
        if os.path.exists(BM25_CACHE):
            print("📦 Loading BM25 corpus from pickle…")
            with open(BM25_CACHE, "rb") as f:
                self.bm25_docs = pickle.load(f)
            tokenized = [d.page_content.lower().split() for d in self.bm25_docs]
            self.bm25 = BM25Okapi(tokenized)
            print(f"✅ BM25 ready with {len(self.bm25_docs)} docs.")
        else:
            raw = self.vectorstore.get()
            texts = raw["documents"]
            metas = raw["metadatas"]
            self.bm25_docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metas)]
            self.bm25 = BM25Okapi([t.lower().split() for t in texts])

    def _rerank(self, query: str, candidates: list[tuple], top_n: int = TOP_K) -> list[tuple[Document, float]]:
        if not candidates:
            return []
        docs, boosts = zip(*candidates)
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        final_scores = [s + b for s, b in zip(scores, boosts)]
        ranked = sorted(zip(docs, final_scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def _retrieve_and_rerank_internal(self, query: str, 
                                      metadata_filter: dict | None = None) -> tuple[list[Document], float, float]:
        source_hint = section_filter_hint(query)
        
        # 1. Similarity Search with Scores
        if metadata_filter:
            sem_pairs = self.vectorstore.similarity_search_with_relevance_scores(query, k=CANDIDATE_K, filter=metadata_filter)
        else:
            sem_pairs = self.vectorstore.similarity_search_with_relevance_scores(query, k=CANDIDATE_K)
            
        # 2. BM25 Search
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:CANDIDATE_K]
        
        candidates = []
        seen = set()
        
        for doc, sim in sem_pairs:
            candidates.append((doc, 0.0, sim))
            seen.add(doc.page_content[:120])
            
        for i in top_idx:
            doc = self.bm25_docs[i]
            if doc.page_content[:120] not in seen:
                candidates.append((doc, 0.0, 0.5)) # Fallback sim for BM25
                
        # 3. Apply Metadata Boosts & Filters
        safe_candidates = []
        blocked_patterns = [
            "ignore previous", "system prompt", "forget your instru", 
            "act as", "you are now", "instead of", "bypass"
        ]
        
        for doc, boost, sim in candidates:
            # 🛡️ Prompt Injection Defense: Purge malicious document content
            content_lower = doc.page_content.lower()
            if any(p in content_lower for p in blocked_patterns):
                continue
                
            if "ignore" in content_lower and "instruction" in content_lower:
                continue
                
            if source_hint and doc.metadata.get("source") == source_hint:
                boost += 0.15
            safe_candidates.append((doc, boost, sim))
            
        if not safe_candidates:
            return [], 0.0, 0.0
            
        # 4. Final Rerank
        docs_only = [d for d, b, s in safe_candidates]
        boosts_only = [b for d, b, s in safe_candidates]
        rerank_pairs = self._rerank(query, list(zip(docs_only, boosts_only)))
        
        top_docs = [d for d, s in rerank_pairs]
        avg_rerank = sum(s for d, s in rerank_pairs) / len(rerank_pairs)
        norm_rerank = (avg_rerank + 5) / 10 # heuristic normalization
        
        # Calculate average similarity for top docs only
        top_sims = []
        for d in top_docs:
            for od, ob, os in safe_candidates:
                if d.page_content == od.page_content:
                    top_sims.append(os)
                    break
        avg_sim = sum(top_sims) / len(top_sims) if top_sims else 0.5
        
        return top_docs, norm_rerank, avg_sim

    def query(self, user_query: str, metadata_filter: dict | None = None) -> dict:
        start_time = time.time()

        # Step 0: Injection Check
        from verification.confidence import detect_injection
        if detect_injection(user_query):
            return {"answer": "Security: Malicious prompt attempt detected.", "confidence": 0.0}

        # Step 1: Rewrite (SMART)
        retrieval_q, intent, display_label = rewrite_query(user_query, llm=self.llm)

        # Step 2: Retrieve (FIRST PASS)
        best_docs, avg_rerank, avg_sim = self._retrieve_and_rerank_internal(retrieval_q, metadata_filter)

        if not best_docs:
            return {"answer": "No relevant info found.", "confidence": 0.0}

        # Step 3: LLM Draft (FIRST PASS)
        context_text = "\n\n---\n\n".join(d.page_content for d in best_docs)
        draft_prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nAnswer:"
        answer = self.llm.invoke(draft_prompt)

        # Step 4: Hybrid Confidence Check
        confidence = compute_confidence(answer, best_docs, reranker_score=avg_rerank, embedding_sim=avg_sim)

        # Step 5: Multi-Hop (UPGRADE: Retrieve -> Group -> Combine)
        # If confidence is low or it's a comparison query, go for multi-hop synthesis
        if confidence < 0.45 or intent == "COMPARISON":
            from retrieval.multi_hop import multi_hop_retrieve, process_multi_hop
            
            # Retrieve Step (Second pass)
            best_docs = multi_hop_retrieve(
                user_query, answer, confidence, 
                lambda q: self._retrieve_and_rerank_internal(q, metadata_filter)[0],
                best_docs
            )
            
            # Group & Combine Step
            print(f"🧠 Synthesizing multi-hop answer for '{intent}' intent...")
            answer = process_multi_hop(user_query, best_docs, self.llm)
            
            # Recalculate confidence for synthesized answer
            confidence = compute_confidence(answer, best_docs, reranker_score=avg_rerank, embedding_sim=avg_sim)

        # Step 6: Gate
        if confidence < 0.5:
            return {
                "answer": "⚠️ Answer not found in documents",
                "confidence": confidence,
                "metrics": {"latency": round(time.time() - start_time, 2)}
            }

        # Step 7: Verify (First Pass)
        verification = verify_answer(answer, best_docs, query=user_query)
        
        # Step 8: Verifier Upgrade - Regeneration if failed (Step 9)
        # If score < 0.7 and NOT a refusal, try to fix it once
        if verification.get("score", 0) < 0.7 and verification.get("score", 0) > 0:
            print(f"⚠️ Low verification ({verification.get('score')}). Regenerating for high-fidelity...")
            regen_prompt = (
                f"Your previous answer was partially ungrounded. Rewrite it STRICTLY using only the provided context. "
                f"Remove any claims not explicitly stated.\n\nContext:\n{context_text}\n\nQuestion: {user_query}\n\nAnswer:"
            )
            answer = self.llm.invoke(regen_prompt)
            verification = verify_answer(answer, best_docs, query=user_query)

        # Step 9: Advanced Metrics (Step 10)
        # Context Precision: Ratio of cited docs in top-K
        cited_indices = {s["doc_idx"] for s in verification.get("supported_sentences", [])}
        context_precision = len(cited_indices) / len(best_docs) if best_docs else 0.0
        
        sources = list({doc.metadata.get("source", "unknown") for doc in best_docs})
        followups = self._generate_followups(answer)
        
        return {
            "answer": answer,
            "docs": best_docs,
            "confidence": round(confidence, 2),
            "verification": verification,
            "intent": intent,
            "sources": sources,
            "followups": followups,
            "metrics": {
                "latency": round(time.time() - start_time, 2),
                "rerank_score": round(avg_rerank, 2),
                "sim_score": round(avg_sim, 2),
                "context_precision": round(context_precision, 2),
                "answer_relevance": verification.get("relevance", 0.0)
            }
        }

    def _generate_followups(self, answer: str) -> list[str]:
        prompt = f"Suggest 2 brief follow-up questions for: {answer}\nReturn only the questions, one per line."
        try:
            response = self.llm.invoke(prompt)
            return [l.strip() for l in response.split("\n") if len(l.strip()) > 5][:3]
        except:
            return ["Tell me more", "What are the rules?"]

    def query_stream(self, user_query: str, metadata_filter: dict | None = None) -> Iterator[str]:
        retrieval_q, _, _ = rewrite_query(user_query, llm=self.llm)
        best_docs, _, _ = self._retrieve_and_rerank_internal(retrieval_q, metadata_filter)
        context_text = "\n\n---\n\n".join(d.page_content for d in best_docs)
        prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nAnswer:"
        yield from self.llm.stream(prompt)
