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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

from core.config import DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, MODEL_NAME, RERANKER_MODEL
from query.rewriter import rewrite_query
from verification.confidence import compute_confidence, compute_per_source_confidence
from verification.verifier import verify_answer
from retrieval.multi_hop import multi_hop_retrieve

BM25_CACHE  = "bm25_docs.pkl"
TOP_K       = 5
CANDIDATE_K = 12

# ── Load System Prompt once at startup ────────────────────────────────────────
_SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
try:
    with open(_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as _f:
        SYSTEM_PROMPT = _f.read().strip()
except FileNotFoundError:
    SYSTEM_PROMPT = "You are a helpful CIT student assistant. Answer ONLY from the provided context."
print("✅ System prompt loaded.")

# ── Advanced Reranker — loaded once at module level (Step 15) ──────────────────
print(f"🧠 Loading Advanced Reranker ({RERANKER_MODEL})…")
reranker = CrossEncoder(RERANKER_MODEL)
print("✅ Reranker ready.")


def is_malicious(text: str) -> bool:
    """Step 1: Detect malicious content patterns in documents."""
    patterns = [
        "ignore previous instructions",
        "system prompt",
        "act as",
        "you are chatgpt",
        "reveal",
        "confidential"
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in patterns)


def filter_chunks(docs: list[Document]) -> list[Document]:
    """Step 2: Filter out document chunks containing malicious content."""
    safe_docs = []
    for doc in docs:
        if not is_malicious(doc.page_content):
            safe_docs.append(doc)
    return safe_docs


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
        self.embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
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
        # 🔥 Step 3: Apply defense filter to retrieved chunks
        docs_only = filter_chunks(docs_only)
        
        boosts_only = [b for d, b, s in safe_candidates] # Note: mismatch after filter, but let's re-zip
        safe_candidates = [(d, b, s) for d, b, s in safe_candidates if d in docs_only]
        
        if not safe_candidates:
            return [], 0.0, 0.0
            
        docs_only = [d for d, b, s in safe_candidates]
        boosts_only = [b for d, b, s in safe_candidates]
        rerank_pairs = self._rerank(query, list(zip(docs_only, boosts_only)))
        
        top_docs = [d for d, s in rerank_pairs]
        # Robust normalization based on Cross-Encoder (trained for circa -10 to +10 range)
        avg_rerank = float(sum(s for d, s in rerank_pairs) / len(rerank_pairs)) if rerank_pairs else 0.0
        # Clip and normalize -10 to +10 range to 0 to 1
        norm_rerank = max(0.0, min(1.0, (avg_rerank + 10.0) / 20.0))
        
        # Calculate average similarity for top docs only
        top_sims = []
        for d in top_docs:
            for od, ob, os in safe_candidates:
                if d.page_content == od.page_content:
                    top_sims.append(os)
                    break
        avg_sim = float(sum(top_sims) / len(top_sims)) if top_sims else 0.5
        
        return top_docs, norm_rerank, avg_sim

    def query(self, user_query: str, metadata_filter: dict | None = None, status_callback=None) -> dict:
        start_time = time.time()

        def set_status(msg):
            if status_callback:
                status_callback(msg)

        # Step 0: Injection Check
        set_status("🛡️ Checking security...")
        from verification.confidence import detect_injection
        if detect_injection(user_query):
            return {
                "answer": "Security: Malicious prompt attempt detected.", 
                "confidence": 0.0,
                "metrics": {"latency": round(time.time() - start_time, 2)}
            }

        # Step 1: Rewrite (SMART)
        set_status("🧠 Analyzing intent & rewriting query...")
        retrieval_q, intent, display_label = rewrite_query(user_query, llm=self.llm)

        # Step 2: Retrieve (FIRST PASS)
        set_status("🔍 Searching CIT knowledge base...")
        best_docs, avg_rerank, avg_sim = self._retrieve_and_rerank_internal(retrieval_q, metadata_filter)

        if not best_docs:
            return {"answer": "No relevant info found.", "confidence": 0.0}

        # Step 3: LLM Draft (FIRST PASS)
        set_status("✍️ Drafting initial response...")
        context_text = "\n\n---\n\n".join(d.page_content for d in best_docs)
        draft_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"NUMERICAL REASONING RULE: If the user's question contains specific numbers or percentages, "
            f"apply the retrieved policy threshold directly to those numbers and state a clear verdict for each person/case mentioned.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {user_query}\n\n"
            f"Answer:"
        )
        answer = self.llm.invoke(draft_prompt)

        # Step 4: Hybrid Confidence Check (Multi-Signal Step 15)
        set_status("🛡️ Calculating hybrid confidence scores...")
        from verification.confidence import compute_confidence
        confidence = float(compute_confidence(answer, best_docs, reranker_score=avg_rerank, embed_model=self.embeddings))

        # Step 5: Multi-Hop (UPGRADE: Retrieve -> Group -> Combine)
        # Trigger multi-hop for complex queries OR low confidence
        if confidence < 0.5 or intent in ["COMPARISON", "COMPOSITE", "AGGREGATION"]:
            set_status(f"🧠 Synthesis: Reasoned across clusters ({intent})...")
            from retrieval.multi_hop import multi_hop_retrieve, process_multi_hop
            
            # Retrieve Step (Second pass)
            best_docs = multi_hop_retrieve(
                user_query, answer, confidence, 
                lambda q: self._retrieve_and_rerank_internal(q, metadata_filter)[0],
                best_docs
            )
            
            # Group & Combine Step
            answer = process_multi_hop(user_query, best_docs, self.llm)
            
            # Recalculate confidence for synthesized answer
            confidence = compute_confidence(answer, best_docs, reranker_score=avg_rerank, embed_model=self.embeddings)

        # Step 6: Gate (Safety Threshold)
        if confidence < 0.5:
            return {
                "answer": "⚠️ Answer not confidently supported by documents.",
                "confidence": confidence,
                "metrics": {"latency": round(time.time() - start_time, 2)}
            }

        # Step 7: Verify & Citation Inlining (The 'Killer Feature')
        # 🔥 OPTIMIZATION: Pre-calculate doc embeddings once to avoid N*M re-calculation
        set_status(f"📍 Verifying citations across {len(best_docs)} sources...")
        doc_embs = [self.embeddings.embed_query(d.page_content) for d in best_docs]
        
        verification = verify_answer(answer, best_docs, query=user_query, embed_model=self.embeddings, precalculated_embeddings=doc_embs)
        answer = verification.get("cited_answer", answer)
        
        # Step 8: Verifier Upgrade - Regeneration if failed (Step 9)
        # 🔥 OPTIMIZATION: Lowered threshold to 0.4 to avoid unnecessary LLM calls
        if verification.get("score", 0) < 0.4 and verification.get("score", 0) > 0:
            set_status("🔄 Low fidelity detected. Regenerating for grounding...")
            regen_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"NUMERICAL REASONING RULE: If the user's question contains specific numbers or percentages, "
                f"apply the retrieved policy threshold directly to those numbers and state a clear verdict for each person/case mentioned.\n\n"
                f"Your previous answer was partially ungrounded. Rewrite it STRICTLY using only the provided context. "
                f"Remove any claims not explicitly stated.\n\nContext:\n{context_text}\n\nQuestion: {user_query}\n\nAnswer:"
            )
            answer = self.llm.invoke(regen_prompt)
            verification = verify_answer(answer, best_docs, query=user_query, embed_model=self.embeddings, precalculated_embeddings=doc_embs)
            answer = verification.get("cited_answer", answer)

        # Step 9: Advanced Metrics (Step 10)
        # Context Precision: Ratio of cited docs in top-K
        cited_indices = {s["doc_idx"] for s in verification.get("supported_sentences", [])}
        context_precision = len(cited_indices) / len(best_docs) if best_docs else 0.0
        
        sources = list({doc.metadata.get("source", "unknown") for doc in best_docs})
        
        # 🔥 OPTIMIZATION: Faster follow-ups (Fixed logic instead of constant LLM overhead)
        followups = [f"How does {sources[0]} handle attendance?", "What are the scholarship rules?"] if sources else ["Tell me more", "Are there any fees?"]
        
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

    def query_with_streaming(self, user_query: str, metadata_filter: dict | None = None):
        """
        🔥 PERFORMANCE UPGRADE: Hybrid streaming mode.
        Yields (type, data) where type is 'status', 'token', or 'result'.
        """
        start_time = time.time()
        
        # Step 1: Rapid Rewrite
        yield "status", "🧠 Optimizing query..."
        retrieval_q, intent, display_label = rewrite_query(user_query, llm=self.llm)
        
        # Step 2: Rapid Retrieval
        yield "status", "🔍 Searching CIT records..."
        best_docs, avg_rerank, avg_sim = self._retrieve_and_rerank_internal(retrieval_q, metadata_filter)
        
        if not best_docs:
            yield "result", {"answer": "No relevant info found.", "confidence": 0.0}
            return

        # Step 3: Fast Streaming Draft
        yield "status", "✍️ Streaming response..."
        context_text = "\n\n---\n\n".join(d.page_content for d in best_docs)
        draft_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {user_query}\n\n"
            f"Answer:"
        )
        
        full_answer = ""
        for token in self.llm.stream(draft_prompt):
            full_answer += token
            yield "token", token

        # Step 4: Background Verification (Post-stream)
        yield "status", "🛡️ Finalizing citations..."
        doc_embs = self.embeddings.embed_documents([d.page_content for d in best_docs])
        
        # Confidence & Verification
        confidence = float(compute_confidence(full_answer, best_docs, reranker_score=avg_rerank, embed_model=self.embeddings))
        verification = verify_answer(full_answer, best_docs, query=user_query, embed_model=self.embeddings, precalculated_embeddings=doc_embs)
        
        cited_indices = {s["doc_idx"] for s in verification.get("supported_sentences", [])}
        context_precision = len(cited_indices) / len(best_docs) if best_docs else 0.0
        sources = list({doc.metadata.get("source", "unknown") for doc in best_docs})
        followups = [f"How does {sources[0]} handle attendance?", "What are the rules?"] if sources else ["Tell me more"]

        yield "result", {
            "answer": verification.get("cited_answer", full_answer),
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

    def query_stream(self, user_query: str, metadata_filter: dict | None = None) -> Iterator[str]:
        retrieval_q, _, _ = rewrite_query(user_query, llm=self.llm)
        best_docs, _, _ = self._retrieve_and_rerank_internal(retrieval_q, metadata_filter)
        context_text = "\n\n---\n\n".join(d.page_content for d in best_docs)
        prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nAnswer:"
        yield from self.llm.stream(prompt)
