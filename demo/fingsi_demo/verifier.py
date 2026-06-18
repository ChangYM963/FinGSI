"""Conservative audit-only verifier for the public demo."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


class ConservativeVerifier:
    """Checks candidate-only and simple consistency conditions."""

    def verify(self, fact: Dict[str, Any], candidates: Sequence[Dict[str, Any]], selected: str) -> Dict[str, Any]:
        candidate_map = {candidate["concept_id"]: candidate for candidate in candidates}
        if selected not in candidate_map:
            return {"passed": False, "flags": ["selected concept is not in candidate pool"]}

        candidate = candidate_map[selected]
        flags: List[str] = []
        if candidate.get("abstract", False):
            flags.append("selected concept is abstract")
        if candidate.get("period_type") and candidate.get("period_type") != fact.get("period_type"):
            flags.append("period type mismatch")
        if fact.get("unit") and fact.get("unit") not in candidate.get("unit_types", []):
            flags.append("unit type mismatch")

        return {
            "passed": not flags,
            "flags": flags,
            "policy": "audit_only_keep_selection",
        }
