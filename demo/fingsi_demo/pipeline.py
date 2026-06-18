"""End-to-end public FinGSI demo pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .graph import LocalDisclosureGraphBuilder, summarize_graph
from .reranker import FeatureComputer, GNNStyleReranker
from .retriever import CandidateRetriever
from .selectors import selector_registry
from .verifier import ConservativeVerifier


def run_case(fact: Dict[str, Any], k: int, selector_name: str) -> Dict[str, Any]:
    candidates = CandidateRetriever(k=k).retrieve(fact)
    graph = LocalDisclosureGraphBuilder().build(fact, candidates)
    features = FeatureComputer(fact, candidates)

    gnn_ranked = GNNStyleReranker().rank(features)
    gnn_scores = {item.concept_id: item.score for item in gnn_ranked}
    selectors = selector_registry()
    verifier = ConservativeVerifier()

    selected_outputs: Dict[str, Any] = {}
    selector_names = list(selectors) if selector_name == "all" else [selector_name]
    for name in selector_names:
        selected = selectors[name].select(features, gnn_scores)  # type: ignore[attr-defined]
        selected_outputs[name] = {
            "selected_concept": selected.concept_id,
            "score": round(selected.score, 4),
            "signals": {key: round(value, 4) for key, value in selected.signals.items()},
            "verifier": verifier.verify(fact, candidates, selected.concept_id),
        }

    return {
        "sample_id": fact.get("sample_id"),
        "line_item": fact.get("line_item"),
        "statement": fact.get("statement"),
        "candidate_pool": [candidate["concept_id"] for candidate in candidates],
        "local_graph": summarize_graph(graph),
        "gnn_reranker_top3": [item.to_json() for item in gnn_ranked[:3]],
        "selectors": selected_outputs,
    }


def run_demo(path: Path, k: int, selector_name: str) -> List[Dict[str, Any]]:
    facts = json.loads(path.read_text(encoding="utf-8"))
    return [run_case(fact, k=k, selector_name=selector_name) for fact in facts]
