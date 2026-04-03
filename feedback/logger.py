"""
Feedback Logger
================
Stores per-answer thumbs-up / thumbs-down ratings in a JSONL file.

Each line is a JSON record:
  {
    "timestamp":  "2026-04-03T08:30:00",
    "query":      "what is the attendance requirement?",
    "answer":     "The minimum attendance requirement is 75%...",
    "confidence": 0.82,
    "verified":   true,
    "rating":     "up",             # "up" | "down"
    "sources":    ["attendance_rules.txt", "CIT_Academic_Calendar.txt"]
  }

This log is the foundation for:
  - Manual review of poor answers
  - Future RLHF / fine-tuning
  - Auto-evaluation dashboards (see analytics.py)
"""

from __future__ import annotations
import json
import os
from datetime import datetime

FEEDBACK_DIR  = "feedback"
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback_log.jsonl")


def log_feedback(
    query: str,
    answer: str,
    confidence: float,
    verified: bool,
    rating: str,          # "up" or "down"
    sources: list[str] | None = None,
) -> None:
    """
    Append a feedback record to the JSONL log.

    Parameters
    ----------
    query      : str   — original user query
    answer     : str   — generated answer
    confidence : float — confidence score (0.0–1.0)
    verified   : bool  — verifier result
    rating     : str   — "up" or "down"
    sources    : list  — source filenames used for the answer
    """
    os.makedirs(FEEDBACK_DIR, exist_ok=True)

    record = {
        "timestamp":  datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "query":      query,
        "answer":     answer[:500],   # truncate very long answers
        "confidence": round(confidence, 3),
        "verified":   verified,
        "rating":     rating,
        "sources":    sources or [],
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    emoji = "👍" if rating == "up" else "👎"
    print(f"📝 Feedback logged: {emoji}  query='{query[:40]}...'")


def load_feedback() -> list[dict]:
    """Load all feedback records from the JSONL log."""
    if not os.path.exists(FEEDBACK_FILE):
        return []
    records = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records
