"""GNN-style structure reranking for the public demo."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .schema import CandidateScore, clamp, lexical_overlap, minmax


class FeatureComputer:
    def __init__(self, fact: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> None:
        self.fact = fact
        self.candidates = list(candidates)
        self.query = " ".join(
            [
                fact.get("line_item", ""),
                fact.get("statement", ""),
                " ".join(fact.get("context_rows", [])),
            ]
        )
        self.retriever_norm = minmax(float(c.get("retriever_score", 0.0)) for c in self.candidates)
        self.structure_norm = minmax(float(c.get("structure_score", 0.0)) for c in self.candidates)

    def features(self, idx: int) -> Dict[str, float]:
        candidate = self.candidates[idx]
        candidate_text = " ".join(
            [
                candidate.get("concept_id", ""),
                candidate.get("label", ""),
                candidate.get("definition", ""),
            ]
        )
        lexical = lexical_overlap(self.query, candidate_text)
        statement_match = self._statement_match(candidate)
        period_match = 1.0 if candidate.get("period_type") == self.fact.get("period_type") else 0.0
        unit_match = 1.0 if self.fact.get("unit") in candidate.get("unit_types", []) else 0.0
        non_abstract = 0.0 if candidate.get("abstract", False) else 1.0
        taxonomy_evidence = clamp(len(candidate.get("taxonomy_neighbors", [])) / 4.0)

        return {
            "lexical": lexical,
            "retriever": self.retriever_norm[idx],
            "structure_prior": self.structure_norm[idx],
            "statement_match": statement_match,
            "period_match": period_match,
            "unit_match": unit_match,
            "non_abstract": non_abstract,
            "taxonomy_evidence": taxonomy_evidence,
        }

    def _statement_match(self, candidate: Dict[str, Any]) -> float:
        statement = self.fact.get("statement", "").lower()
        roles = [role.lower() for role in candidate.get("statement_roles", [])]
        if not roles:
            return 0.0
        return 1.0 if any(role in statement or statement in role for role in roles) else 0.0


class GNNStyleReranker:
    """Transparent stand-in for the paper's relation-aware GNN reranker."""

    def rank(self, features: FeatureComputer) -> List[CandidateScore]:
        scored: List[CandidateScore] = []
        for idx, candidate in enumerate(features.candidates):
            signals = features.features(idx)
            score = (
                0.30 * signals["structure_prior"]
                + 0.18 * signals["statement_match"]
                + 0.14 * signals["period_match"]
                + 0.12 * signals["unit_match"]
                + 0.12 * signals["lexical"]
                + 0.08 * signals["taxonomy_evidence"]
                + 0.06 * signals["retriever"]
            )
            scored.append(CandidateScore(candidate["concept_id"], score, signals))
        return sorted(scored, key=lambda item: item.score, reverse=True)
