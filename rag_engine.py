"""
🚀 Unified RAG Engine — Advanced v3
======================================
Full pipeline:
  User Query
    │ rewrite_query()          ← LLM-powered + rule-based fallback
    ▼
  Hybrid Search                ← Vector (top-20) + BM25 full-corpus (top-20)
    ▼
  Metadata Boost               ← HOD hit? +0.2 · source match? +0.15
    ▼
  Cross-Encoder Reranking      ← ms-marco-MiniLM-L-6-v2
    ▼
  Top 5 chunks
    ▼
  Multi-step fallback          ← if empty → re-search with broader query
    ▼
  LLM Draft (gemma3:1b)
    ▼
  Multi-hop (if conf < 0.45)   ← entity extraction → second retrieval pass
    ▼
  Final LLM Answer
    ▼
  Multi-factor Confidence      ← keyword overlap + length + citation
    ▼
  Confidence gate              ← conf < 0.3 → polite "not found" reply
"""

from __future__ import annotations
import os
import pickle
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


# ── Section filter hint ───────────────────────────────────────────────────────
def section_filter_hint(query: str) -> str | None:
    """Return a source-file hint when the intent is unambiguous."""
    q = query.lower()
    if any(w in q for w in ["hod", "head of department", "faculty"]):
        return "CIT_Academic_Calendar.txt"
    if "attendance" in q:
        return "attendance_rules.txt"
    if "scholarship" in q:
        return "scholarship.txt"
    if "hostel" in q:
        return "hostel_rules.txt"
    if "exam" in q or "fail" in q or "reappear" in q:
        return "exam_policy.txt"
    if "placement" in q:
        return "placement_policy.txt"
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

    # ── BM25 ─────────────────────────────────────────────────────────────────
    def _load_bm25(self):
        if os.path.exists(BM25_CACHE):
            print("📦 Loading BM25 corpus from pickle…")
            with open(BM25_CACHE, "rb") as f:
                self.bm25_docs = pickle.load(f)
            tokenized = [d.page_content.lower().split() for d in self.bm25_docs]
            self.bm25 = BM25Okapi(tokenized)
            print(f"✅ BM25 ready with {len(self.bm25_docs)} docs.")
        else:
            print("⚠️  bm25_docs.pkl not found — building BM25 from Chroma…")
            raw = self.vectorstore.get()
            texts = raw["documents"]
            metas = raw["metadatas"]
            self.bm25_docs = [
                Document(page_content=t, metadata=m)
                for t, m in zip(texts, metas)
            ]
            self.bm25 = BM25Okapi([t.lower().split() for t in texts])
            print(f"✅ BM25 fallback ready with {len(texts)} docs.")

    # ── Raw retrieval with metadata boost ────────────────────────────────────
    def _retrieve_candidates(self, expanded_q: str, source_hint: str | None,
                              metadata_filter: dict | None = None):
        # 1. Vector search
        if metadata_filter:
            sem_results = self.vectorstore.similarity_search(
                expanded_q, k=CANDIDATE_K, filter=metadata_filter
            )
        else:
            sem_results = self.vectorstore.similarity_search(expanded_q, k=CANDIDATE_K)

        # 2. BM25
        tokens      = expanded_q.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_idx     = sorted(range(len(bm25_scores)),
                             key=lambda i: bm25_scores[i], reverse=True)[:CANDIDATE_K]
        bm25_results = [self.bm25_docs[i] for i in top_idx]

        # 3. Merge & deduplicate
        seen, candidates = set(), []
        for doc in sem_results + bm25_results:
            key = doc.page_content[:120]
            if key not in seen:
                seen.add(key)
                candidates.append(doc)

        # 4. Metadata boost
        boosted = []
        q_lower = expanded_q.lower()
        for doc in candidates:
            boost = 0.0
            content_lower = doc.page_content.lower()
            if "hod" in q_lower and "head of department" in content_lower:
                boost += 0.2
            if source_hint and doc.metadata.get("source", "") == source_hint:
                boost += 0.15
            boosted.append((doc, boost))

        return boosted

    # ── Cross-Encoder reranking ───────────────────────────────────────────────
    def _rerank(self, query: str, candidates: list[tuple], top_n: int = TOP_K):
        if not candidates:
            return []
        docs, boosts = zip(*candidates)
        pairs        = [(query, doc.page_content) for doc in docs]
        scores       = reranker.predict(pairs)
        final_scores = [s + b for s, b in zip(scores, boosts)]
        ranked       = sorted(zip(docs, final_scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]

    # ── Public retrieval (used by multi-hop as a callable) ───────────────────
    def _retrieve_and_rerank_raw(self, query: str,
                                  metadata_filter: dict | None = None) -> list:
        source_hint = section_filter_hint(query)
        candidates  = self._retrieve_candidates(query, source_hint, metadata_filter)
        top_docs    = self._rerank(query, candidates)

        if not top_docs:
            print("⚠️  Empty results — retrying with broadened query…")
            broader     = query + " CIT policy rules information"
            candidates  = self._retrieve_candidates(broader, None, None)
            top_docs    = self._rerank(broader, candidates)

        return top_docs

    # ── Full Q&A pipeline (sync) ─────────────────────────────────────────────
    def query(self, user_query: str,
              metadata_filter: dict | None = None) -> dict:
        """
        Run the full RAG pipeline and return a rich result dict.
        """
        import time
        start_time = time.time()

        # ── Step 1: Rewrite query & Detect Intent ────────────────────────────
        retrieval_q, intent, display_label = rewrite_query(user_query, llm=self.llm)

        # ── Step 2: Retrieval Strategy based on Intent ───────────────────────
        # For procedural/comparison, we widen the context net initially
        current_k = TOP_K
        if intent in ["PROCEDURAL", "COMPARISON"]:
            current_k = TOP_K + 3 

        best_docs = self._retrieve_and_rerank_raw(retrieval_q, metadata_filter)

        # ── Step 3: Build context & draft answer ─────────────────────────────
        system_prompt = "You are a CIT student assistant."
        if intent == "COMPARISON":
            system_prompt += (
                " You MUST provide a clear comparison using a Markdown table. "
                "Each row should represent a feature (e.g., Fees, Duration, Eligibility)."
            )
        elif intent == "PROCEDURAL":
            system_prompt += " Provide a clear step-by-step process using a numbered list."

        if os.path.exists("system_prompt.txt"):
            with open("system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()

        context_text = "\n\n---\n\n".join(d.page_content for d in best_docs)
        draft_prompt = (
            f"{system_prompt}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {user_query}\n\n"
            f"Answer:"
        )
        draft_answer = self.llm.invoke(draft_prompt)

        # ── Step 4: Confidence check → multi-hop if needed ───────────────────
        draft_conf = compute_confidence(draft_answer, best_docs)
        
        # Multi-hop is ALWAYS used for Comparison to ensure we didn't miss one side
        force_hop = (intent == "COMPARISON")
        
        if draft_conf < 0.45 or force_hop:
            best_docs = multi_hop_retrieve(
                original_query=retrieval_q,
                draft_answer=draft_answer,
                confidence=draft_conf,
                retrieve_fn=lambda q: self._retrieve_and_rerank_raw(q, metadata_filter),
                existing_docs=best_docs,
            )

        # ── Step 5: Final answer (re-generate if docs changed) ───────────────
        if len(best_docs) > TOP_K:
            context_text = "\n\n---\n\n".join(d.page_content for d in best_docs)
            final_prompt = (
                f"{system_prompt}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {user_query}\n\n"
                f"Answer:"
            )
            answer = self.llm.invoke(final_prompt)
        else:
            answer = draft_answer

        # ── Step 6: Confidence gate ───────────────────────────────────────────
        final_conf = compute_confidence(answer, best_docs)
        if final_conf < 0.3 and "not found" not in answer.lower():
            answer = (
                "I could not find this information clearly in the documents. "
                "Please contact the college office or refer to the official CIT handbook."
            )

        # ── Step 7: Verification & Metadata (Advanced Metrics) ───────────────
        verification = verify_answer(answer, best_docs, query=user_query)
        per_src_conf = compute_per_source_confidence(answer, best_docs)
        sources      = list({doc.metadata.get("source", "unknown") for doc in best_docs})
        
        # ── Step 8: Follow-up Suggestions ────────────────────────────────────
        followups = self._generate_followups(answer)
        
        latency = time.time() - start_time

        return {
            "answer":          answer,
            "docs":            best_docs,
            "confidence":      round(final_conf, 2),
            "per_src_conf":    per_src_conf,
            "verification":    verification,
            "rewritten_query": display_label,
            "intent":          intent,
            "sources":         sources,
            "followups":       followups,
            "metrics":         {
                "latency": round(latency, 2),
                "tokens":  len(answer.split()) * 1.3, # estimated
            }
        }

    def _generate_followups(self, answer: str) -> list[str]:
        """Generate 2-3 brief follow-up questions."""
        prompt = (
            "You are a CIT student assistant. Based on the answer provided, suggest 2 or 3 brief follow-up questions "
            "that the student might want to ask next. Keep them under 10 words. "
            "Return ONLY the list of questions, one per line. No numbers.\n\n"
            f"Answer: {answer}\n\n"
            "Follow-up Questions:"
        )
        try:
            response = self.llm.invoke(prompt)
            return [line.strip().strip("-*") for line in response.split("\n") if len(line.strip()) > 5][:3]
        except:
            return ["Tell me more about this", "What are the rules?"]

    # ── Streaming query ──────────────────────────────────────────────────────
    def query_stream(self, user_query: str,
                     metadata_filter: dict | None = None) -> Iterator[str]:
        """
        Stream version: Returns iterator. Note: Multi-hop not applied in stream.
        """
        retrieval_q, intent, _ = rewrite_query(user_query, llm=self.llm)
        best_docs = self._retrieve_and_rerank_raw(retrieval_q, metadata_filter)

        system_prompt = "You are a CIT student assistant."
        context_text = "\n\n---\n\n".join(d.page_content for d in best_docs)
        prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {user_query}\n\nAnswer:"
        
        yield from self.llm.stream(prompt)
