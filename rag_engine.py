"""
🚀 Unified RAG Engine — Antigravity v2
=========================================
Pipeline:
  User Query
    │ expand_query()          ← acronym + context injection
    │ section_filter_hint()   ← file-scope hints for structured data
    ▼
  Hybrid Search              ← Vector (top-20) + BM25 full-corpus (top-20)
    ▼
  Metadata Boost             ← HOD hit? +0.2 score bump
    ▼
  Cross-Encoder Reranking    ← ms-marco-MiniLM-L-6-v2
    ▼
  Top 5 chunks only
    ▼
  Multi-step fallback        ← if empty → re-search with wider expansion
    ▼
  LLM (gemma3:1b via Ollama)
    ▼
  Confidence gate            ← if conf < 0.3 → polite "not found" reply
"""

import os
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from core.config import DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, MODEL_NAME

BM25_CACHE   = "bm25_docs.pkl"
TOP_K        = 5           # ← fixed at 5 (not 8 or 10)
CANDIDATE_K  = 20          # candidates per source before rerank

# ── Reranker (loaded once at module level) ────────────────────────────────────
print("🧠 Loading Cross-Encoder reranker…")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("✅ Reranker ready.")


# ── LEVEL 1 · Step 4 : Query Expansion ───────────────────────────────────────
def expand_query(query: str) -> str:
    """Expand abbreviations + inject topical context keywords."""
    q = query.lower().strip()

    acronyms = {
        "ai & ds":  "artificial intelligence and data science",
        "ai ds":    "artificial intelligence and data science",
        "ai&ds":    "artificial intelligence and data science",
        "aids":     "artificial intelligence and data science",
        "hod":      "head of department",
        "cse":      "computer science and engineering",
        "ece":      "electronics and communication engineering",
        "eee":      "electrical and electronics engineering",
        "it dept":  "information technology",
        "mca":      "master of computer applications",
        "mba":      "master of business administration",
        "cgpa":     "cumulative grade point average",
        "sgpa":     "semester grade point average",
        "backlog":  "arrear failed subject reappear",
        "arrears":  "arrear failed subject reappear",
    }
    for key, val in acronyms.items():
        if key in q:
            q += " " + val

    # Exact-match override (Level 4 · Step 12)
    if "hod" in q:
        q += " head of department faculty name location"
    if "attendance" in q:
        q += " minimum 75 percent requirement eligibility shortfall"
    if "scholarship" in q:
        q += " eligibility application process renewal"
    if "placement" in q:
        q += " eligibility arrear backlog criteria package"
    if "exam" in q or "fail" in q:
        q += " reappear supplementary arrear policy"
    if "hostel" in q:
        q += " rules timings curfew visitor"
    if "fee" in q or "fees" in q:
        q += " amount payment penalty late"

    # Slang / informal normalization
    slang = {
        "bro":          "",
        "any chance":   "is it possible eligibility",
        "what to do":   "procedure process steps",
        "now what":     "next steps process",
        "came late":    "late entry curfew timing",
        "low attendance":"attendance shortage below 75",
        "attendance short": "attendance shortage below 75",
    }
    for key, val in slang.items():
        if key in q:
            q = q.replace(key, val)

    return q.strip()


# ── LEVEL 4 · Step 11 : Section-Based Retrieval Hint ─────────────────────────
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
        self.embeddings   = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore  = Chroma(
            persist_directory=DB_DIR,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
        )
        self.llm = OllamaLLM(model=MODEL_NAME, temperature=0)
        self._load_bm25()

    # ── BM25 from full-corpus pickle ──────────────────────────────────────────
    def _load_bm25(self):
        if os.path.exists(BM25_CACHE):
            print("📦 Loading BM25 corpus from pickle…")
            with open(BM25_CACHE, "rb") as f:
                self.bm25_docs = pickle.load(f)
            tokenized = [d.page_content.lower().split() for d in self.bm25_docs]
            self.bm25 = BM25Okapi(tokenized)
            print(f"✅ BM25 ready with {len(self.bm25_docs)} full-corpus docs.")
        else:
            # Fallback: pull from Chroma (acceptable if pickle not yet built)
            print("⚠️  bm25_docs.pkl not found — building BM25 from Chroma…")
            raw = self.vectorstore.get()
            texts = raw["documents"]
            metas = raw["metadatas"]
            from langchain_core.documents import Document
            self.bm25_docs = [
                Document(page_content=t, metadata=m)
                for t, m in zip(texts, metas)
            ]
            self.bm25 = BM25Okapi([t.lower().split() for t in texts])
            print(f"✅ BM25 fallback ready with {len(texts)} docs.")

    # ── Core retrieval ────────────────────────────────────────────────────────
    def _retrieve_candidates(self, expanded_q: str, source_hint: str | None):
        """Return up to CANDIDATE_K*2 candidate (doc, raw_score) pairs."""
        # 1. Vector search
        sem_results = self.vectorstore.similarity_search(expanded_q, k=CANDIDATE_K)

        # 2. BM25 keyword search
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

        # 4. LEVEL 2 · Step 6 : Metadata Boost  ──────────────────────────────
        boosted = []
        q_lower  = expanded_q.lower()
        for doc in candidates:
            boost = 0.0
            content_lower = doc.page_content.lower()
            # HOD boost
            if "hod" in q_lower and "head of department" in content_lower:
                boost += 0.2
            # Source-file preference
            if source_hint and doc.metadata.get("source", "") == source_hint:
                boost += 0.15
            boosted.append((doc, boost))

        return boosted

    # ── Cross-Encoder Reranking (Level 1 · Step 1) ───────────────────────────
    def _rerank(self, query: str, candidates: list[tuple], top_n: int = TOP_K):
        if not candidates:
            return []
        docs, boosts = zip(*candidates)
        pairs  = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        final_scores = [s + b for s, b in zip(scores, boosts)]
        ranked = sorted(zip(docs, final_scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]

    # ── Multi-Step Fallback (Level 4 · Step 10) ──────────────────────────────
    def retrieve_and_rerank(self, query: str) -> list:
        expanded_q   = expand_query(query)
        source_hint  = section_filter_hint(query)

        candidates = self._retrieve_candidates(expanded_q, source_hint)
        top_docs   = self._rerank(query, candidates)

        # Fallback: if reranker returned nothing, widen search
        if not top_docs:
            print("⚠️  Empty results — retrying with broadened query…")
            broader_q  = expanded_q + " CIT policy rules information"
            candidates = self._retrieve_candidates(broader_q, source_hint=None)
            top_docs   = self._rerank(broader_q, candidates)

        return top_docs

    # ── Full Q&A pipeline ─────────────────────────────────────────────────────
    def query(self, user_query: str) -> tuple[str, list]:
        best_docs = self.retrieve_and_rerank(user_query)

        # Level 2 · Step 2 : top-5 only  (context already limited by TOP_K)
        context_text = "\n\n---\n\n".join(d.page_content for d in best_docs)

        # Load system prompt
        system_prompt = "You are a CIT student assistant."
        if os.path.exists("system_prompt.txt"):
            with open("system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()

        full_prompt = (
            f"{system_prompt}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {user_query}\n\n"
            f"Answer:"
        )
        answer = self.llm.invoke(full_prompt)

        # Level 3 · Step 9 : Confidence gate (low conf → polite fallback)
        from verification.confidence import compute_confidence
        conf = compute_confidence(answer, best_docs)
        if conf < 0.3 and "not found" not in answer.lower():
            answer = (
                "I could not find this information clearly in the documents. "
                "Please contact the college office or refer to the official CIT handbook."
            )

        return answer, best_docs
