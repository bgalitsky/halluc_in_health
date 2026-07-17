#!/usr/bin/env python3
"""Run feature-group ablations for leakage and robustness checks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

# Reuse functions from run_evaluation.py when scripts are run from package root or scripts dir.
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from run_evaluation import evaluate_one, feature_columns  # noqa: E402


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else config_path.parent / p


def drop_prefix(cols: List[str], prefixes: tuple[str, ...]) -> List[str]:
    return [c for c in cols if not c.startswith(prefixes)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    features_path = resolve_path(config_path, args.features or cfg["outputs"]["features_csv"])
    output_path = resolve_path(config_path, args.output or cfg["outputs"]["ablations_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_path)
    all_cols = feature_columns(df)

    ablations = {
        "full": all_cols,
        "lexical_style_only": [c for c in all_cols if c.startswith("lex_")],
        "tree_topology_only": [c for c in all_cols if c.startswith("top_")],
        "discourse_relation_labels_only": [c for c in all_cols if c.startswith("disc_") or c.startswith("role_")],
        "nucleus_satellite_only": [c for c in all_cols if c.startswith("ns_")],
        "domain_cues_only": [c for c in all_cols if c.startswith("domain_")],
        "discourse_only": [c for c in all_cols if c.startswith(("top_", "disc_", "role_", "ns_"))],
        "jkrhm_without_discourse": [c for c in all_cols if c.startswith("jkrhm_") and c != "jkrhm_disc_score"],
        "jkrhm_plus_discourse": [c for c in all_cols if c.startswith(("jkrhm_", "top_", "disc_", "role_", "ns_"))],
        "no_lexical_style": drop_prefix(all_cols, ("lex_",)),
        "no_domain_cues": drop_prefix(all_cols, ("domain_",)),
        "no_nucleus_satellite": drop_prefix(all_cols, ("ns_",)),
        "no_relation_labels": drop_prefix(all_cols, ("disc_", "role_")),
        "no_topology": drop_prefix(all_cols, ("top_",)),
    }

    rows = []
    for name, cols in ablations.items():
        if not cols:
            continue
        for seed in cfg["model"]["random_seeds"]:
            rows.append(evaluate_one(df, cols, cfg, int(seed), name))
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote ablation metrics to {output_path}")


if __name__ == "__main__":
    main()
