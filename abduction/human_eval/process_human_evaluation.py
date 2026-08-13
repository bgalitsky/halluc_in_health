"""
Process human-evaluation logs for Chapter 9.

Usage:
    python process_human_evaluation.py \
        --input human_evaluation_log.csv \
        --out-dir human_eval_results

Outputs:
    system_summary.csv
    dataset_summary.csv
    participant_group_summary.csv
    human_judgment_metrics.csv
    quality_checks.csv
    overall_summary.json

Notes
-----
- confidence_before / confidence_after are recorded on a 0–100 scale and
  normalized to [0,1].
- chapter9_confidence_delta = mean(post - pre), reproducing the simple
  before/after confidence difference described in Chapter 9.
- correctness_aware_gain is an additional diagnostic:
    + confidence increase when the final judgment is correct,
    - confidence increase when the final judgment is incorrect.
"""

import argparse
import json
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "participant_id", "participant_role", "stimulus_id", "dataset", "case_id",
    "system", "gold_hallucination", "initial_judgment", "confidence_before",
    "final_judgment", "confidence_after", "clarity", "coherence", "trust"
]

def parse_bool_series(s: pd.Series) -> pd.Series:
    truthy = {"1", "true", "t", "yes", "y", "hallucinated"}
    falsy = {"0", "false", "f", "no", "n", "not hallucinated", "non-hallucinated"}
    out = []
    for value in s:
        if isinstance(value, bool):
            out.append(value)
            continue
        text = str(value).strip().lower()
        if text in truthy:
            out.append(True)
        elif text in falsy:
            out.append(False)
        else:
            raise ValueError(f"Cannot parse Boolean value: {value!r}")
    return pd.Series(out, index=s.index, dtype=bool)

def judgment_to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("hallucinated")

def classification_metrics(gold: pd.Series, pred: pd.Series) -> dict:
    gold = gold.astype(bool)
    pred = pred.astype(bool)
    tp = int((gold & pred).sum())
    fp = int((~gold & pred).sum())
    fn = int((gold & ~pred).sum())
    tn = int((~gold & ~pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "n": tp + fp + fn + tn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()
    d["gold_bool"] = parse_bool_series(d["gold_hallucination"])
    d["initial_pred"] = judgment_to_bool(d["initial_judgment"])
    d["final_pred"] = judgment_to_bool(d["final_judgment"])

    numeric = ["confidence_before", "confidence_after",
               "clarity", "coherence", "trust"]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["confidence_before_norm"] = d["confidence_before"] / 100.0
    d["confidence_after_norm"] = d["confidence_after"] / 100.0
    d["chapter9_confidence_delta"] = (
        d["confidence_after_norm"] - d["confidence_before_norm"]
    )
    d["initial_correct"] = (d["initial_pred"] == d["gold_bool"]).astype(int)
    d["final_correct"] = (d["final_pred"] == d["gold_bool"]).astype(int)

    # Additional correctness-aware calibration diagnostic.
    raw_shift = d["chapter9_confidence_delta"]
    d["correctness_aware_gain"] = raw_shift.where(
        d["final_correct"].eq(1), -raw_shift
    )

    return d

def aggregate_ratings(d: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = (
        d.groupby(group_col, dropna=False)
         .agg(
             n=("stimulus_id", "size"),
             evaluators=("participant_id", "nunique"),
             clarity_mean=("clarity", "mean"),
             coherence_mean=("coherence", "mean"),
             trust_mean=("trust", "mean"),
             confidence_before_mean=("confidence_before_norm", "mean"),
             confidence_after_mean=("confidence_after_norm", "mean"),
             chapter9_confidence_delta=("chapter9_confidence_delta", "mean"),
             correctness_aware_gain=("correctness_aware_gain", "mean"),
             initial_accuracy=("initial_correct", "mean"),
             final_accuracy=("final_correct", "mean"),
         )
         .reset_index()
    )
    return out

def grouped_human_metrics(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, g in d.groupby("system", dropna=False):
        initial = classification_metrics(g["gold_bool"], g["initial_pred"])
        final = classification_metrics(g["gold_bool"], g["final_pred"])
        rows.append({
            "system": system,
            "n": len(g),
            "initial_precision": initial["precision"],
            "initial_recall": initial["recall"],
            "initial_f1": initial["f1"],
            "initial_accuracy": initial["accuracy"],
            "final_precision": final["precision"],
            "final_recall": final["recall"],
            "final_f1": final["f1"],
            "final_accuracy": final["accuracy"],
            "f1_change": final["f1"] - initial["f1"],
            "accuracy_change": final["accuracy"] - initial["accuracy"],
        })
    return pd.DataFrame(rows)

def quality_checks(d: pd.DataFrame) -> pd.DataFrame:
    checks = []
    duplicate_count = int(
        d.duplicated(subset=["participant_id", "stimulus_id"], keep=False).sum()
    )
    checks.append(("duplicate_participant_stimulus_rows", duplicate_count))

    for col in ["clarity", "coherence", "trust"]:
        bad = int((~d[col].between(1, 5) | d[col].isna()).sum())
        checks.append((f"{col}_outside_1_5_or_missing", bad))

    for col in ["confidence_before", "confidence_after"]:
        bad = int((~d[col].between(0, 100) | d[col].isna()).sum())
        checks.append((f"{col}_outside_0_100_or_missing", bad))

    checks.append(("missing_required_values",
                   int(d[REQUIRED_COLUMNS].isna().any(axis=1).sum())))
    return pd.DataFrame(checks, columns=["check", "count"])

def main(input_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    d = prepare(raw)

    system_summary = aggregate_ratings(d, "system")
    dataset_summary = aggregate_ratings(d, "dataset")
    group_summary = aggregate_ratings(d, "participant_role")
    human_metrics = grouped_human_metrics(d)
    qc = quality_checks(d)

    system_summary.to_csv(out / "system_summary.csv", index=False)
    dataset_summary.to_csv(out / "dataset_summary.csv", index=False)
    group_summary.to_csv(out / "participant_group_summary.csv", index=False)
    human_metrics.to_csv(out / "human_judgment_metrics.csv", index=False)
    qc.to_csv(out / "quality_checks.csv", index=False)

    overall_initial = classification_metrics(d["gold_bool"], d["initial_pred"])
    overall_final = classification_metrics(d["gold_bool"], d["final_pred"])

    summary = {
        "n_rows": int(len(d)),
        "n_evaluators": int(d["participant_id"].nunique()),
        "datasets": sorted(map(str, d["dataset"].dropna().unique())),
        "systems": sorted(map(str, d["system"].dropna().unique())),
        "mean_clarity": float(d["clarity"].mean()),
        "mean_coherence": float(d["coherence"].mean()),
        "mean_trust": float(d["trust"].mean()),
        "mean_confidence_before": float(d["confidence_before_norm"].mean()),
        "mean_confidence_after": float(d["confidence_after_norm"].mean()),
        "chapter9_mean_confidence_delta": float(
            d["chapter9_confidence_delta"].mean()
        ),
        "mean_correctness_aware_gain": float(
            d["correctness_aware_gain"].mean()
        ),
        "initial_human_judgment": overall_initial,
        "final_human_judgment": overall_final,
    }
    (out / "overall_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Wrote:")
    for p in sorted(out.iterdir()):
        print(" -", p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV evaluation log")
    parser.add_argument("--out-dir", default="human_eval_results")
    args = parser.parse_args()
    main(args.input, args.out_dir)
