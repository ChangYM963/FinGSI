"""Leakage-safe local disclosure graph skeleton for the public demo."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


class LocalDisclosureGraphBuilder:
    """Builds a small local graph from candidate-visible information only."""

    def build(self, fact: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        nodes: List[Dict[str, str]] = [
            {"id": "target", "type": "target", "text": fact.get("line_item", "")},
            {"id": "statement", "type": "statement", "text": fact.get("statement", "")},
            {"id": "unit", "type": "unit", "text": fact.get("unit", "")},
            {"id": "period_type", "type": "period_type", "text": fact.get("period_type", "")},
        ]
        edges: List[Dict[str, str]] = [
            {"source": "target", "target": "statement", "type": "appears_in_statement"},
            {"source": "target", "target": "unit", "type": "has_unit"},
            {"source": "target", "target": "period_type", "type": "has_period_type"},
        ]

        for idx, row in enumerate(fact.get("context_rows", []), start=1):
            row_id = f"context:{idx}"
            nodes.append({"id": row_id, "type": "context_row", "text": row})
            edges.append({"source": "target", "target": row_id, "type": "nearby_row"})

        seen_nodes = {node["id"] for node in nodes}
        candidate_ids = {candidate["concept_id"] for candidate in candidates}

        for candidate in candidates:
            concept_id = candidate["concept_id"]
            nodes.append(
                {
                    "id": concept_id,
                    "type": "candidate_concept",
                    "text": candidate.get("label", concept_id),
                }
            )
            seen_nodes.add(concept_id)
            edges.append({"source": "target", "target": concept_id, "type": "candidate"})

        for candidate in candidates:
            concept_id = candidate["concept_id"]
            for role in candidate.get("statement_roles", []):
                role_id = f"role:{role}"
                if role_id not in seen_nodes:
                    nodes.append({"id": role_id, "type": "taxonomy_role", "text": role})
                    seen_nodes.add(role_id)
                edges.append({"source": concept_id, "target": role_id, "type": "has_statement_role"})

            for neighbor in candidate.get("taxonomy_neighbors", []):
                neighbor_id = neighbor.get("concept_id", "unknown")
                if neighbor_id not in seen_nodes and neighbor_id not in candidate_ids:
                    nodes.append(
                        {
                            "id": neighbor_id,
                            "type": neighbor.get("relation", "taxonomy_neighbor"),
                            "text": neighbor.get("label", neighbor_id),
                        }
                    )
                    seen_nodes.add(neighbor_id)
                edges.append(
                    {
                        "source": concept_id,
                        "target": neighbor_id,
                        "type": neighbor.get("relation", "taxonomy_relation"),
                    }
                )

        return {"nodes": nodes, "edges": edges}


def summarize_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    node_types: Dict[str, int] = {}
    edge_types: Dict[str, int] = {}
    for node in graph["nodes"]:
        node_types[node["type"]] = node_types.get(node["type"], 0) + 1
    for edge in graph["edges"]:
        edge_types[edge["type"]] = edge_types.get(edge["type"], 0) + 1
    return {
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "node_types": node_types,
        "edge_types": edge_types,
    }
