"""
LLM-Powered Query Rewriter
============================
Converts informal / vague student queries into formal retrieval-optimized queries.

Examples:
  "when is deadline?"    → "Scholarship application deadline eligibility criteria 2024"
  "bro attendance low"  → "Attendance shortage below 75 percent consequences policy"
  "hod details"         → "Head of department faculty name contact location"

Falls back to rule-based expansion if the LLM rewrite fails or produces garbage.
"""

from __future__ import annotations


_REWRITE_PROMPT = """You are a search query optimizer for a college student information system.
Convert the following informal student question into a formal, specific, retrieval-optimized search query.

Rules:
- Expand abbreviations (HOD → head of department, CGPA → cumulative grade point average, etc.)
- Make the intent explicit and specific
- Add relevant college/academic keywords
- Keep it under 15 words
- Return ONLY the rewritten query, nothing else. No explanation.

Student question: {question}
Rewritten query:"""


def rewrite_query(question: str, llm=None) -> tuple[str, str]:
    """
    Rewrite a student question into a retrieval-optimized query.

    Parameters
    ----------
    question : str        — raw user input
    llm      : OllamaLLM  — if None, falls back to rule-based expansion

    Returns
    -------
    (retrieval_query, display_label)
      retrieval_query  — used for actual vector/BM25 search
      display_label    — shown in the UI as "I interpreted your question as: ..."
    """
    base_expanded = _rule_based_expand(question)

    if llm is None:
        return base_expanded, _friendly_label(question)

    try:
        prompt = _REWRITE_PROMPT.format(question=question)
        response = llm.invoke(prompt)
        rewritten = _clean_rewrite(response, question)

        if len(rewritten.split()) < 2 or len(rewritten) > 200:
            return base_expanded, _friendly_label(question)

        # Merge: LLM rewrite + rule-based keyword injection for robustness
        merged = rewritten + " " + _extra_keywords(question)
        return merged.strip(), rewritten

    except Exception as e:
        print(f"⚠️  Query rewriter LLM call failed ({e}), using rule-based fallback.")
        return base_expanded, _friendly_label(question)


def _clean_rewrite(response: str, original: str) -> str:
    """Strip quotes, newlines, and common prefixes from LLM output."""
    text = response.strip()
    for prefix in ["rewritten query:", "query:", "answer:", "search query:", "optimized query:"]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.strip('"\'')
    text = text.split("\n")[0].strip()
    return text if text else original


def _friendly_label(q: str) -> str:
    """Create a UI-friendly interpretation label from rule-based expansion."""
    q = q.strip()
    return q[0].upper() + q[1:] if q else q


def _extra_keywords(q: str) -> str:
    """Inject domain keywords based on detected intent."""
    q_lower = q.lower()
    extras = []
    if any(w in q_lower for w in ["hod", "head", "faculty", "professor", "staff"]):
        extras.append("faculty name contact location department CIT")
    if "attendance" in q_lower:
        extras.append("minimum 75 percent requirement shortage condonation")
    if "scholarship" in q_lower:
        extras.append("eligibility application renewal merit income criteria")
    if any(w in q_lower for w in ["exam", "fail", "backlog", "arrear", "supplementary"]):
        extras.append("reappear supplementary arrear policy exam rules")
    if "hostel" in q_lower:
        extras.append("rules curfew timing visitor warden")
    if any(w in q_lower for w in ["fee", "fees", "payment"]):
        extras.append("amount due date penalty late payment structure")
    if "placement" in q_lower:
        extras.append("eligibility criteria package company arrear backlog")
    if "deadline" in q_lower:
        extras.append("last date submission deadline academic calendar")
    return " ".join(extras)


def _rule_based_expand(query: str) -> str:
    """Rule-based query expansion — always runs as the base layer."""
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

    slang = {
        "bro":              "",
        "any chance":       "is it possible eligibility",
        "what to do":       "procedure process steps",
        "now what":         "next steps process",
        "came late":        "late entry curfew timing",
        "low attendance":   "attendance shortage below 75",
        "attendance short": "attendance shortage below 75",
    }
    for key, val in slang.items():
        if key in q:
            q = q.replace(key, val)

    q += " " + _extra_keywords(query)
    return q.strip()
