"""Readable and learned-fusion selector analogues for the public demo."""

from __future__ import annotations

from typing import Dict, List

from .reranker import FeatureComputer
from .schema import CandidateScore, minmax


class GraphSummaryPromptSelector:
    """Readable graph-to-text prompt analogue."""

    name = "graph_summary_prompt"

    def select(self, features: FeatureComputer, gnn_scores: Dict[str, float]) -> CandidateScore:
        del gnn_scores
        scored: List[CandidateScore] = []
        for idx, candidate in enumerate(features.candidates):
            signals = features.features(idx)
            score = (
                0.44 * signals["lexical"]
                + 0.24 * signals["retriever"]
                + 0.16 * signals["statement_match"]
                + 0.10 * signals["period_match"]
                + 0.06 * signals["unit_match"]
            )
            scored.append(CandidateScore(candidate["concept_id"], score, signals))
        return max(scored, key=lambda item: item.score)


class GNNConditionedPromptSelector:
    """Readable prompt that includes GNN ranks and scores."""

    name = "gnn_conditioned_prompt"

    def select(self, features: FeatureComputer, gnn_scores: Dict[str, float]) -> CandidateScore:
        gnn_norm_map = normalize_score_map(gnn_scores)
        scored: List[CandidateScore] = []
        for idx, candidate in enumerate(features.candidates):
            signals = features.features(idx)
            gnn = gnn_norm_map[candidate["concept_id"]]
            score = (
                0.38 * gnn
                + 0.24 * signals["lexical"]
                + 0.18 * signals["retriever"]
                + 0.12 * signals["statement_match"]
                + 0.08 * signals["period_match"]
            )
            enriched = dict(signals)
            enriched["gnn_score"] = gnn
            scored.append(CandidateScore(candidate["concept_id"], score, enriched))
        return max(scored, key=lambda item: item.score)


class SingleTokenFusionSelector:
    """One pooled structure-token analogue."""

    name = "single_token_fusion"

    def select(self, features: FeatureComputer, gnn_scores: Dict[str, float]) -> CandidateScore:
        pooled_structure = sum(gnn_scores.values()) / max(len(gnn_scores), 1)
        scored: List[CandidateScore] = []
        for idx, candidate in enumerate(features.candidates):
            signals = features.features(idx)
            score = (
                0.30 * signals["lexical"]
                + 0.22 * signals["retriever"]
                + 0.20 * signals["structure_prior"]
                + 0.18 * pooled_structure
                + 0.10 * signals["statement_match"]
            )
            enriched = dict(signals)
            enriched["pooled_structure_token"] = pooled_structure
            scored.append(CandidateScore(candidate["concept_id"], score, enriched))
        return max(scored, key=lambda item: item.score)


class MultiTokenFusionSelector:
    """Candidate-level structure-token analogue."""

    name = "multi_token_fusion"

    def select(self, features: FeatureComputer, gnn_scores: Dict[str, float]) -> CandidateScore:
        gnn_norm_map = normalize_score_map(gnn_scores)
        scored: List[CandidateScore] = []
        for idx, candidate in enumerate(features.candidates):
            signals = features.features(idx)
            candidate_token = gnn_norm_map[candidate["concept_id"]]
            score = (
                0.32 * candidate_token
                + 0.24 * signals["lexical"]
                + 0.18 * signals["retriever"]
                + 0.14 * signals["statement_match"]
                + 0.12 * signals["period_match"]
            )
            enriched = dict(signals)
            enriched["candidate_structure_token"] = candidate_token
            scored.append(CandidateScore(candidate["concept_id"], score, enriched))
        return max(scored, key=lambda item: item.score)


def normalize_score_map(scores: Dict[str, float]) -> Dict[str, float]:
    keys = list(scores)
    vals = minmax(scores[key] for key in keys)
    return {key: vals[idx] for idx, key in enumerate(keys)}


def selector_registry() -> Dict[str, object]:
    selectors = [
        GraphSummaryPromptSelector(),
        GNNConditionedPromptSelector(),
        SingleTokenFusionSelector(),
        MultiTokenFusionSelector(),
    ]
    return {selector.name: selector for selector in selectors}
