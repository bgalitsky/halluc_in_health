#!/usr/bin/env python3
"""Summarize repeated-split metrics with mean, standard deviation, and 95% CIs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:  # pragma: no cover
    ttest_rel = None
    wilcoxon = None


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else config_path.parent / p


def summarize(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method"):
        row = {"method": method, "n_runs": len(g)}
        for m in metrics:
            if m not in g:
                continue
            vals = g[m].astype(float).to_numpy()
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci = 1.96 * std / np.sqrt(max(len(vals), 1))
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95_low"] = mean - ci
            row[f"{m}_ci95_high"] = mean + ci
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_mean" if "f1_mean" in rows[0] else "method", ascending=False)


def paired_tests(df: pd.DataFrame, baseline: str, challenger: str, metric: str = "f1") -> Dict[str, Any]:
    a = df[df["method"] == baseline].sort_values("seed")[["seed", metric]]
    b = df[df["method"] == challenger].sort_values("seed")[["seed", metric]]
    merged = a.merge(b, on="seed", suffixes=("_baseline", "_challenger"))
    out: Dict[str, Any] = {"baseline": baseline, "challenger": challenger, "metric": metric, "n_pairs": len(merged)}
    if len(merged) < 2:
        return out
    x = merged[f"{metric}_baseline"].astype(float).to_numpy()
    y = merged[f"{metric}_challenger"].astype(float).to_numpy()
    if ttest_rel is not None:
        stat, p = ttest_rel(y, x, nan_policy="omit")
        out["paired_t_stat"] = float(stat)
        out["paired_t_p"] = float(p)
    if wilcoxon is not None:
        try:
            stat, p = wilcoxon(y, x)
            out["wilcoxon_stat"] = float(stat)
            out["wilcoxon_p"] = float(p)
        except Exception:
            pass
    out["mean_delta"] = float(np.nanmean(y - x))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None, help="Metrics CSV. Defaults to ablations if present, else metrics.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tests-output", default="results/paired_tests.csv")
    parser.add_argument("--baseline", default="jkrhm_without_discourse")
    parser.add_argument("--challenger", default="jkrhm_plus_discourse")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    default_input = cfg["outputs"].get("ablations_csv")
    input_path = resolve_path(config_path, args.input or default_input)
    if not input_path.exists():
        input_path = resolve_path(config_path, cfg["outputs"]["metrics_csv"])
    output_path = resolve_path(config_path, args.output or cfg["outputs"]["summary_csv"])
    tests_path = resolve_path(config_path, args.tests_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tests_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    metrics = [m for m in ["accuracy", "precision", "recall", "f1", "auroc", "tpr_at_fpr_10", "tpr_at_fpr_5"] if m in df.columns]
    summary = summarize(df, metrics)
    summary.to_csv(output_path, index=False)
    tests = paired_tests(df, args.baseline, args.challenger, metric="f1")
    pd.DataFrame([tests]).to_csv(tests_path, index=False)
    print(f"Wrote summary to {output_path}")
    print(f"Wrote paired tests to {tests_path}")


if __name__ == "__main__":
    main()
