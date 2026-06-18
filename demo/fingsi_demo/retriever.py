"""Candidate retrieval stage for the public demo."""

from __future__ import annotations

from typing import Any, Dict, List


class CandidateRetriever:
    """Fixed candidate-pool stage.

    The paper uses a deterministic CandidateRetriever with top-50 candidates.
    This demo starts from a small candidate list in JSON and ranks it by the
    provided retriever score.
    """

    def __init__(self, k: int) -> None:
        self.k = k

    def retrieve(self, fact: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = list(fact.get("candidates", []))
        return sorted(
            candidates,
            key=lambda item: float(item.get("retriever_score", 0.0)),
            reverse=True,
        )[: self.k]
