"""Shared data structures and text helpers for the public demo."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "of", "or", "the", "to", "used",
}


def tokenize(text: str) -> List[str]:
    return [tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]


def minmax(values: Iterable[float]) -> List[float]:
    vals = list(values)
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        return [0.5 for _ in vals]
    return [(value - lo) / (hi - lo) for value in vals]


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def lexical_overlap(query: str, candidate_text: str) -> float:
    query_tokens = set(tokenize(query))
    candidate_tokens = set(tokenize(candidate_text))
    if not query_tokens or not candidate_tokens:
        return 0.0
    score = len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)

    query_lower = query.lower()
    candidate_lower = candidate_text.lower()
    phrase_bonuses = [
        ("financing", 0.10),
        ("investing", 0.10),
        ("operating", 0.10),
        ("equity", 0.08),
        ("net income", 0.10),
        ("common stock", 0.08),
    ]
    for phrase, bonus in phrase_bonuses:
        if phrase in query_lower and phrase in candidate_lower:
            score += bonus
    return clamp(score)


@dataclass
class CandidateScore:
    concept_id: str
    score: float
    signals: Dict[str, float]

    def to_json(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "score": round(self.score, 4),
            "signals": {key: round(value, 4) for key, value in self.signals.items()},
        }
