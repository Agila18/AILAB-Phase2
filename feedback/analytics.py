"""
Feedback Analytics
====================
Reads feedback_log.jsonl and computes useful statistics for system improvement.

Functions:
  summary()          → overall up/down counts, average confidence
  failing_queries()  → queries rated 👎 sorted by frequency
  source_stats()     → average confidence per source file
  low_confidence()   → all answered with conf < threshold + 👎 rating
"""

from __future__ import annotations
from collections import defaultdict
from feedback.logger import load_feedback


def summary() -> dict:
    """Return high-level feedback statistics."""
    records = load_feedback()
    if not records:
        return {"total": 0, "thumbs_up": 0, "thumbs_down": 0, "avg_confidence": 0.0}

    up   = sum(1 for r in records if r.get("rating") == "up")
    down = sum(1 for r in records if r.get("rating") == "down")
    avg_conf = round(sum(r.get("confidence", 0) for r in records) / len(records), 3)

    return {
        "total":          len(records),
        "thumbs_up":      up,
        "thumbs_down":    down,
        "avg_confidence": avg_conf,
        "approval_rate":  round(up / len(records), 2) if records else 0.0,
    }


def failing_queries(top_n: int = 10) -> list[dict]:
    """Return the most frequently 👎-rated query patterns."""
    records = load_feedback()
    counts: dict[str, int] = defaultdict(int)

    for r in records:
        if r.get("rating") == "down":
            # Normalise query to catch duplicates
            key = r.get("query", "").lower().strip()
            counts[key] += 1

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [{"query": q, "count": c} for q, c in sorted_items[:top_n]]


def source_stats() -> list[dict]:
    """Return average confidence broken down by source document."""
    records = load_feedback()
    source_conf: dict[str, list[float]] = defaultdict(list)

    for r in records:
        for src in r.get("sources", []):
            source_conf[src].append(r.get("confidence", 0.0))

    result = []
    for src, confs in source_conf.items():
        result.append({
            "source":      src,
            "avg_conf":    round(sum(confs) / len(confs), 3),
            "num_answers": len(confs),
        })

    return sorted(result, key=lambda x: x["avg_conf"])


def low_confidence_failures(threshold: float = 0.5) -> list[dict]:
    """Return 👎-rated answers that also had low confidence — most actionable failures."""
    records = load_feedback()
    return [
        r for r in records
        if r.get("rating") == "down" and r.get("confidence", 1.0) < threshold
    ]
