#!/usr/bin/env python3
"""Command-line entry point for the public FinGSI demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fingsi_demo.pipeline import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public FinGSI framework demo.")
    parser.add_argument("samples", type=Path, help="Path to samples.json")
    parser.add_argument("--k", type=int, default=5, help="Candidate pool size for the demo")
    parser.add_argument(
        "--selector",
        choices=[
            "all",
            "graph_summary_prompt",
            "gnn_conditioned_prompt",
            "single_token_fusion",
            "multi_token_fusion",
        ],
        default="all",
        help="Selector to run. The default runs every public demo selector.",
    )
    args = parser.parse_args()

    results = run_demo(args.samples, k=args.k, selector_name=args.selector)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
