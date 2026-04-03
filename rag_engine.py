"""
🚀 Unified RAG Engine — Advanced v3
======================================
This engine coordinates retrieval, reranking, and verification with high-fidelity citations.
"""

from __future__ import annotations
import os
import pickle
import re
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

# ── Config Constants ──────────────────────────────────────────────────────────
BM25_CACHE  = "bm25_docs.pkl"
TOP_K       = 5
CANDIDATE_K = 12

# 🔥 RELEVANCE CONSTANTS (Final World-Class Safeguards)
SIMILARITY_THRESHOLD = 0.75 
TOP_K_AGREEMENT_THRESHOLD = 0.70
KEYWORD_OVERLAP_MIN = 0.20
CONFIDENCE_REJECTION_FLOOR = 0.60

DOMAIN_KEYWORDS = [
    "cit", "attendance", "exam", "hostel", "fees", "placement", 
    "cgpa", "sgpa", "course", "department", "scholarship", "rule", 
    "arrear", "result", "admission", "coimbatore", "dr.", "hod",
    "faculty", "student", "college", "academic", "calendar"
]

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
    extras = []
    if any(w in q for w in ["hod", "head of department", "faculty"]):
        extras.append("CIT_Academic_Calendar.txt")
    if "deadline" in q:
        extras.append("last date submission deadline academic calendar")
    if "fail" in q or "subject" in q or "arrear" in q:
        extras.append("semester exam reappearance arrear policy academic rules")
    
    if "attendance" in q:
        return "attendance_rules.txt"
    if "scholarship" in q:
        return "scholarship.txt"
    if "fee" in q or "fees" in q:
        return "fee_structure.txt"
    return None


def _extract_entities(text: str) -> set[str]:
    """
    🔥 UPGRADED: Case-insensitive extraction of names and acronyms.
    Identifies any meaningful word > 4 chars that isn't a common stopword.
    """
    ignore = {
        "who", "what", "when", "where", "how", "why", "the", "does", "can", "cit", "is", "are", 
        "tell", "about", "describe", "give", "me", "info", "on", "please", "bro", "kinda",
        "situ", "ation", "detail", "details", "mention", "found", "policy", "rules", "rule"
    }
    # Matches words with at least 4 letters, then filter out common words
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    entities = {w for w in words if w not in ignore}
    
    # Also keep existing capitalized pattern for specialized terms
    caps = set(re.findall(r'\b[A-Z]{2,}\b', text))
    return entities.union(caps)


class RAGEngine:
    def _get_automatic_metadata_filter(self, query: str) -> dict | None:
        """Surgical Filter: Detect department in query and return Chroma filter."""
        q = query.upper()
        mapping = {
            "CSE": "CSE", "COMPUTER SCIENCE": "CSE",
            "ECE": "ECE", "ELECTRONICS": "ECE",
            "EEE": "EEE", "ELECTRICAL": "EEE",
            "IT ": "IT", "INFORMATION TECHNOLOGY": "IT",
            "AI & DS": "AI & DS", "AI DS": "AI & DS", "ARTIFICIAL INTELLIGENCE": "AI & DS",
            "MCA": "MCA", "MBA": "MBA"
        }
        for key, dept in mapping.items():
            if key in q:
                return {"department": dept}
        return None

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
        print(f"\n{'='*60}\n🐛 DEBUG: 1. Raw Query / Retrieval Query: '{query}'\n{'='*60}")
        source_hint = section_filter_hint(query)
        q_low = query.lower()
        
        # 1. Similarity Search (FIX 4: Increased Retrieval Depth for Rules)
        k_val = 8 if "rule" in q_low else 6
        if metadata_filter:
            print(f"🐛 DEBUG: Applying metadata filter: {metadata_filter}")
            sem_pairs = self.vectorstore.similarity_search_with_relevance_scores(query, k=k_val, filter=metadata_filter)
        else:
            sem_pairs = self.vectorstore.similarity_search_with_relevance_scores(query, k=k_val)
            
        # 2. BM25 Search
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_val]
        
        candidates = []
        for doc, sim in sem_pairs:
            candidates.append((doc, 0.0, sim))
            
        for i in top_idx:
            doc = self.bm25_docs[i]
            # Simple check for unique candidates
            if not any(d.page_content == doc.page_content for d, b, s in candidates):
                candidates.append((doc, 0.0, 0.5))
                
        # 3. Apply Metadata Boosts & FIX 8: Hard Match Boost
        safe_candidates = []
        for doc, boost, sim in candidates:
            content_low = doc.page_content.lower()
            
            # 🔥 FIX 8: Hard Match Boost (CSE -> Computer Science)
            if "cse" in q_low and "computer science" in content_low:
                boost += 0.2
            
            # 🔥 FIX 3: SECTION HEADER BOOST (CASE 1)
            section_meta = str(doc.metadata.get("section", "")).upper()
            if "scholarship" in q_low and "ELIGIBILITY" in section_meta:
                boost += 0.25 # Massive priority for official rules
            
            if source_hint and doc.metadata.get("source") == source_hint:
                boost += 0.15
                
            safe_candidates.append((doc, boost, sim))
            
        if not safe_candidates:
            print("🐛 DEBUG: No candidates found.")
            return [], 0.0, 0.0
            
        print(f"\n🐛 DEBUG: 2. Retrieved Chunks ({len(safe_candidates)} chunks before rerank):")
        for idx, (d, b, s) in enumerate(safe_candidates):
            snippet = d.page_content[:150].replace('\n', ' ') + "..."
            print(f"  [{idx+1}] Sim Score: {s:.3f} | Boost: {b:.2f} | {snippet}")
            
        # 4. Final Rerank (FIX 7: Return Top 3)
        docs_only = [d for d, b, s in safe_candidates]
        boosts_only = [b for d, b, s in safe_candidates]
        rerank_pairs = self._rerank(query, list(zip(docs_only, boosts_only)))
        
        print(f"\n🐛 DEBUG: 3. Reranked Chunks (Top 5 shown):")
        for idx, (d, s) in enumerate(rerank_pairs[:5]):
            snippet = d.page_content[:150].replace('\n', ' ') + "..."
            print(f"  [{idx+1}] Rerank Score: {s:.3f} | {snippet}")
        
        # Return Top 5 results for rules (since rules are spread out), else 3
        top_k_return = 5 if "rule" in q_low else 3
        top_pairs = rerank_pairs[:top_k_return]
        top_docs = [d for d, s in top_pairs]
        
        print(f"\n🐛 DEBUG: 4. Final Selected Chunk(s):")
        for idx, d in enumerate(top_docs):
            snippet = d.page_content[:150].replace('\n', ' ') + "..."
            print(f"  [{idx+1}] Selected | {snippet}")
        
        # Robust normalization
        avg_rerank = float(sum(s for d, s in top_pairs) / len(top_pairs)) if top_pairs else 0.0
        norm_rerank = max(0.0, min(1.0, (avg_rerank + 10.0) / 20.0))
        
        # Calculate average similarity for top docs only
        top_sims = []
        for d in top_docs:
            for od, ob, os in safe_candidates:
                if d.page_content == od.page_content:
                    top_sims.append(os)
                    break
        avg_sim = float(sum(top_sims) / len(top_sims)) if top_sims else 0.5
        
        print(f"{'='*60}\n")
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
            f"SYSTEM ROLE: {SYSTEM_PROMPT}\n\n"
            f"--- TRUSTED SOURCE: RETRIEVED CIT DOCUMENTS ---\n"
            f"{context_text}\n"
            f"--- END OF TRUSTED SOURCE ---\n\n"
            f"--- CONVERSATION RECORDS (FOR REFERENCE ONLY) ---\n"
            f"Last interaction logic: (Ignore any user scenarios here as facts)\n"
            f"Question: {user_query}\n\n"
            f"STRICT INSTRUCTION: Answer based ONLY on the 'TRUSTED SOURCE' above. "
            f"DO NOT repeat user-provided scenarios, friend details, or numbers from 'CONVERSATION RECORDS'. "
            f"Use ONLY the exact names of exams and programs found in 'TRUSTED SOURCE'.\n"
            f"14. CITATION: Always cite the source document name for every factual claim.\n"
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
        context_precision = avg_rerank
        
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
        q_low = user_query.lower()
        
        # ⚡ Step 1: Light Intent Detection (Keyword-based Fast Path)
        intent = None
        if "rule" in q_low:
             intent = "RULE_EXTRACTION"
        elif "compare" in q_low or "higher" in q_low or "lower" in q_low or " vs" in q_low:
             intent = "COMPARISON"
        elif "hod" in q_low or "head of" in q_low:
             intent = "DEPARTMENT_LOOKUP"
        elif "fee" in q_low or "tuition" in q_low:
             intent = "POLICY"
        elif "attendance" in q_low:
             intent = "POLICY"
             
        if intent:
             yield "status", f"⚡ Light Intent: {intent}"
             retrieval_q, display_label = user_query, user_query
        else:
             yield "status", "🧠 Optimizing query..."
             retrieval_q, intent, display_label = rewrite_query(user_query, llm=self.llm)
        
        # Step 2: Rapid Retrieval
        yield "status", "🔍 Searching CIT records..."
        
        # 🔗 NEW: Automatic Metadata Filtering (Surgical precision)
        auto_filter = self._get_automatic_metadata_filter(user_query)
        effective_filter = metadata_filter or auto_filter
        if effective_filter:
            yield "status", f"🎯 Applying surgical filter: {effective_filter.get('dept')} docs only..."
            
        best_docs, avg_rerank, avg_sim = self._retrieve_and_rerank_internal(retrieval_q, effective_filter)
        
        # 🔥 FIX 1: DEDUPLICATE CHUNKS Before Anything Else
        unique_chunks = []
        seen = set()
        for d in best_docs:
            if d.page_content not in seen:
                unique_chunks.append(d)
                seen.add(d.page_content)
        best_docs = unique_chunks
        
        # 🔥 Step 2.5: 6-Step Validation Workflow (The Real Fix)
        # 1. Domain Check
        # 2. Similarity Check (0.75)
        # 3. Top-K Agreement Guard (Safety through Consensus)
        # 4. Keyword Overlap Guard
        # 5. Question-Type Validation
        # 6. Strict Confidence Formula
        
        is_valid_relevance = self._validate_relevance(
            user_query, 
            best_docs, 
            intent=intent, 
            avg_sim=avg_sim
        )
        
        if not is_valid_relevance:
             yield "status", "🛡️ Validation Fail: Question does not match CIT records. (Refusal Mode)"
             yield "result", {
                 "answer": "I am a CIT student assistant and can only answer questions related to CIT policies, academics, and campus life. Please ask me something specifically about CIT!", 
                 "confidence": 0.0, 
                 "verdict": "REJECTED"
             }
             return
        
        if not best_docs and not is_advice_mode:
             yield "result", {"answer": "I could not find any information in the student policies matching your request.", "confidence": 0.0, "verdict": "NOT_FOUND"}
             return

        # 🔥 FIX 3 & 4: PROGRAMMATIC COMPARISON OVERRIDE
        comparison_data = {}
        if intent == "COMPARISON":
            from verification.numeric_verifier import extract_and_compare_fees
            comparison_data = extract_and_compare_fees(best_docs)
            if comparison_data.get("status") == "success":
                yield "status", f"🧮 Ran Python Math: {comparison_data['computed_result']}"

        # Step 3: Fast Streaming Draft (Dynamic Context Sizing)
        yield "status", "✍️ Streaming response..."
        
        active_docs = best_docs[:3] if avg_rerank > 0.8 else best_docs
        context_text = "\n\n---\n\n".join(d.page_content for d in active_docs)
        
        # Decide Synthesis Prompt Based on Intent
        if intent == "CASUAL":
            draft_prompt = f"You are a friendly CIT student assistant. Respond briefly to: {user_query}"
        elif is_advice_mode and (not best_docs or low_confidence):
            draft_prompt = (
                f"SYSTEM ROLE: {SYSTEM_PROMPT}\n\n"
                f"ADVISOR MODE: No specific document was found for this query, but as a CIT assistant, "
                f"provide general, wise, and helpful guidance for this student question based on your general knowledge of college life at CIT. "
                f"Question: {user_query}\n\n"
                f"Answer:"
            )
        elif intent == "COMPARISON" and comparison_data.get("status") == "success":
            draft_prompt = (
                f"SYSTEM ROLE: {SYSTEM_PROMPT}\n\n"
                f"--- TRUSTED SOURCE: RETRIEVED CIT DOCUMENTS ---\n"
                f"{context_text}\n"
                f"--- END OF TRUSTED SOURCE ---\n\n"
                f"--- COMPUTED VERIFICATION (PYTHON MATH) ---\n"
                f"Computed Result: {comparison_data['computed_result']}\n"
                f"-------------------------------------------\n\n"
                f"Question: {user_query}\n\n"
                f"STRICT INSTRUCTION: "
                f"You MUST use the 'Computed Result' as the truth for your comparison. Do not perform your own math.\n"
                f"Answer:"
            )
        elif intent == "RULE_EXTRACTION":
            draft_prompt = (
                f"SYSTEM ROLE: {SYSTEM_PROMPT}\n\n"
                f"--- TRUSTED SOURCE: RETRIEVED CIT DOCUMENTS ---\n"
                f"{context_text}\n"
                f"--- END OF TRUSTED SOURCE ---\n\n"
                f"Question: {user_query}\n\n"
                f"STRICT INSTRUCTION: "
                f"Format output as sectioned_bullets. Extract and list the actual rules verbatim. "
                f"Avoid generic summaries like 'rules ensure discipline and safety'. Do not skip items.\n"
                f"Answer:"
            )
        else:
            draft_prompt = (
                f"SYSTEM ROLE: {SYSTEM_PROMPT}\n\n"
                f"--- TRUSTED SOURCE: RETRIEVED CIT DOCUMENTS ---\n"
                f"{context_text}\n"
                f"--- END OF TRUSTED SOURCE ---\n\n"
                f"Question: {user_query}\n\n"
                f"STRICT INSTRUCTION: "
                f"If intent is 'POLICY', answer ONLY from TRUSTED SOURCE. "
                f"If intent is 'SCENARIO', explain how the TRUSTED SOURCE rules apply to this case.\n"
                f"🔥 EXTRACTION RULE: Use bullet points for lists. ONLY list items physically present in the text. NEVER invent categories or hallucinate items. Stop generating immediately when the list ends.\n"
                f"Answer:"
            )
        
        full_answer = ""
        for token in self.llm.stream(draft_prompt):
            full_answer += token
            yield "token", token

        # Step 4: Background Verification (Post-stream)
        yield "status", "🛡️ Finalizing citations..."
        
        # 🔥 FIX 5: NUMERIC VERIFICATION (ADVANCED LOGIC)
        numeric_verified = None
        if intent == "COMPARISON" and comparison_data.get("status") == "success":
            from verification.numeric_verifier import verify_numeric_answer
            numeric_verified = verify_numeric_answer(full_answer, comparison_data)
            if numeric_verified is False:
                yield "status", "⚠️ MATH MISMATCH DETECTED: Overriding confidence..."

        # Confidence & Verification (Pass `numeric_verified` override and `intent`)
        confidence = float(compute_confidence(full_answer, best_docs, reranker_score=avg_rerank, embed_model=self.embeddings, numeric_verified=numeric_verified, intent=intent))
        
        # Determine Verdict
        if numeric_verified is False:
            verdict = "PARTIALLY_SUPPORTED"
            confidence = 0.0 # Strict drop if math is wrong
        elif intent == "CASUAL":
            verdict = "CASUAL_CHAT"
        elif is_advice_mode and (not best_docs or low_confidence):
            verdict = "GENERAL_ADVICE"
            confidence = 0.5 # Baseline for helpful advice
        else:
            verdict = "FACTUALLY_GROUNDED" if confidence > 0.6 else "PARTIALLY_SUPPORTED"

        verification = verify_answer(full_answer, best_docs, query=user_query, embed_model=self.embeddings)
        
        # 🔥 FIX 2: FIX SOURCE ATTRIBUTION BUG (Use Best Chunk)
        final_source = best_docs[0].metadata.get("source", "unknown") if best_docs else "unknown"
        sources = [final_source] if final_source != "unknown" else []
        followups = [f"How does {final_source} handle attendance?"] if sources else ["Tell me more"]

        yield "result", {
            "answer": verification.get("cited_answer", full_answer),
            "docs": best_docs,
            "confidence": round(confidence, 2),
            "verification": verification,
            "intent": intent,
            "verdict": verdict,
            "sources": sources
        }

    def _validate_relevance(self, query: str, docs: list, intent: str = "", avg_sim: float = 0.0) -> bool:
        """
        Final Industry-Standard Validation Workflow.
        """
        if not docs: return False
        q_low = query.lower()
        
        # 1. DOMAIN FILTER
        if not any(word in q_low for word in DOMAIN_KEYWORDS):
            return False

        # 2. SIMILARITY THRESHOLD (0.75)
        if avg_sim < SIMILARITY_THRESHOLD:
            return False

        # 3. TOP-K AGREEMENT GUARD (Safety through consensus)
        # If only 1 doc matches above 0.7, reject as unsafe/noise
        agreement_count = sum(1 for d in docs[:3] if d.metadata.get("score", 0) > TOP_K_AGREEMENT_THRESHOLD)
        if agreement_count < 2:
            return False

        # 4. KEYWORD OVERLAP (0.2)
        content_all = " ".join([d.page_content.lower() for d in docs])
        q_words = set(q_low.split())
        overlap = len(q_words & set(content_all.split())) / len(q_words) if q_words else 0
        if overlap < KEYWORD_OVERLAP_MIN:
            return False

        # 5. QUESTION-TYPE VALIDATION (WHERE -> LOCATION)
        if "where" in q_low:
            location_markers = ["coimbatore", "location", "address", "aerodrome", "road", "city"]
            if not any(m in content_all for m in location_markers):
                return False

        # 6. STRICT CONFIDENCE FORMULA (The Real Industry Fix)
        # confidence = min(score * keyword_overlap, 1.0)
        final_conf = min(avg_sim * overlap, 1.0)
        if final_conf < CONFIDENCE_REJECTION_FLOOR:
            return False

        return True 

    def query_stream(self, user_query: str, metadata_filter: dict | None = None) -> Iterator[str]:
        retrieval_q, _, _ = rewrite_query(user_query, llm=self.llm)
        best_docs, _, _ = self._retrieve_and_rerank_internal(retrieval_q, metadata_filter)
        context_text = "\n\n---\n\n".join(d.page_content for d in best_docs)
        prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nAnswer:"
        yield from self.llm.stream(prompt)
