#!/usr/bin/env python3
"""
MetaPCR-LLM evaluation.

Computes:
    - Accuracy, precision, recall, hallucination F1
    - Escalation appropriateness
    - Constraint violation rate
    - Expected Calibration Error (ECE)
    - Step Validity Accuracy (SVA)
    - Chain Certification Rate / Accuracy (CCR / CCA)
    - Bridge Validation Accuracy (BVA)
    - Bridge false-accept / false-reject rates
    - Failure Localization Accuracy (FLA)
    - Failure-type localization
    - Validator localization
    - Relative Reasoning Hallucination Rate (RRHR)
    - Dataset-level results
    - Reasoning-regime results
    - Mixed-logic results
    - Ablation results
    - Latency mean / median / p95
    - Human evaluation
    - LaTeX table generation

Published values are stored only as reference baselines.
They are NOT treated as new MetaPCR measurements.

Requirements:
    pip install pandas numpy scikit-learn

Example:

    python metapcr_evaluation.py evaluate \
        --predictions runs/full_metapcr.jsonl \
        --out results/full_metapcr

    python metapcr_evaluation.py evaluate-runs \
        --run-dir runs \
        --out results/ablations.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    brier_score_loss,
    cohen_kappa_score,
)


# ================================================================
# 1. PUBLISHED REFERENCE VALUES
# ================================================================

# ValidLLP4LLM overall evaluation.
# These are inherited reference results, not MetaPCR measurements.

VALIDLLP_OVERALL = pd.DataFrame(
    [
        ["LP",                   0.68, 0.62, 0.59, 0.18, 0.17, 0.21],
        ["LP + Negation + IC",   0.72, 0.69, 0.64, 0.09, 0.15, 0.24],
        ["PLP",                  0.74, 0.71, 0.67, 0.14, 0.11, 0.36],
        ["ALP",                  0.73, 0.70, 0.69, 0.13, 0.14, 0.33],
        ["Argumentation",        0.75, 0.73, 0.71, 0.12, 0.13, 0.41],
        ["DeLP",                 0.76, 0.74, 0.72, 0.11, 0.12, 0.44],
        ["CLP(FD/R)",            0.71, 0.66, 0.63, 0.08, 0.16, 0.29],
        ["Stable-model / ASP",   0.74, 0.70, 0.68, 0.10, 0.14, 0.47],
        ["Epistemic / Modal LP", 0.70, 0.67, 0.66, 0.13, 0.15, 0.39],
        ["Description Logic",    0.69, 0.64, 0.61, 0.10, 0.16, 0.26],
        ["ValidLLP4LLM",         0.83, 0.81, 0.79, 0.06, 0.08, 0.38],
    ],
    columns=[
        "method",
        "accuracy",
        "f1",
        "escalation",
        "violation_rate",
        "calibration_error",
        "time_s",
    ],
)


VALIDLLP_REGIME = pd.DataFrame(
    [
        ["LP",           0.81, 0.58, 0.55, 0.57, 0.66, 0.63],
        ["LP + IC",      0.79, 0.60, 0.54, 0.64, 0.78, 0.67],
        ["PLP",          0.73, 0.82, 0.61, 0.63, 0.68, 0.69],
        ["ALP",          0.70, 0.65, 0.83, 0.62, 0.66, 0.69],
        ["AF / DeLP",    0.72, 0.66, 0.71, 0.84, 0.67, 0.72],
        ["CLP",          0.69, 0.61, 0.56, 0.60, 0.85, 0.66],
        ["ValidLLP4LLM", 0.82, 0.80, 0.81, 0.83, 0.84, 0.82],
    ],
    columns=[
        "method",
        "rule",
        "uncertainty",
        "abduction",
        "conflict",
        "constraint",
        "mean",
    ],
)


VALIDLLP_DATASET_F1 = pd.DataFrame(
    [
        ["Truthful-Halluc",           0.62, 0.67, 0.72, 0.78, 0.85],
        ["Med-Halluc",                0.57, 0.60, 0.77, 0.72, 0.81],
        ["eSNLI-Halluc",              0.58, 0.62, 0.68, 0.67, 0.69],
        ["Autoimmune-narrate-halluc", 0.39, 0.37, 0.32, 0.35, 0.33],
    ],
    columns=[
        "dataset",
        "LP",
        "Argumentation",
        "LLP",
        "LLP+Context",
        "Full LLP",
    ],
)


VALIDLLP_ABLATION = pd.DataFrame(
    [
        ["Full LLP",                     0.83, 0.81, 0.79],
        ["- provenance labels",          0.79, 0.76, 0.73],
        ["- uncertainty labels",         0.78, 0.75, 0.72],
        ["- temporal labels",            0.77, 0.74, 0.71],
        ["- contextual-priority labels", 0.80, 0.77, 0.75],
        ["- defeasibility labels",       0.76, 0.73, 0.70],
    ],
    columns=["variant", "accuracy", "f1", "escalation"],
)


VALIDLLP_HUMAN = pd.DataFrame(
    [
        ["LLM only",          0.67, np.nan, np.nan, 2.3],
        ["+ LP",              0.73, 0.12, 0.54, 3.1],
        ["+ PLP",             0.74, 0.14, 0.56, 3.4],
        ["+ Argumentation",   0.75, 0.16, 0.59, 3.7],
        ["+ Abduction",       0.71, 0.11, 0.51, 3.3],
        ["Full ValidLLP4LLM", 0.83, 0.22, 0.68, 4.2],
    ],
    columns=[
        "condition",
        "human_accuracy",
        "delta_confidence",
        "kappa",
        "interpretability",
    ],
)


VALIDLLP_RRHR = pd.DataFrame(
    [
        ["Truth-o-Meter-style", 0.32],
        ["VERUS-LM",             0.28],
        ["LOGIC-LM",             0.27],
        ["Adaptive Solver Routing", 0.25],
        ["LP Ensemble",          0.22],
    ],
    columns=["method", "rrhr"],
)


# Information-theoretic / abductive reference values.

IG_DATASET_F1 = pd.DataFrame(
    [
        ["TruthfulHalluc", 0.63, 0.66, 0.71, 0.72, 0.79, 0.86],
        ["MedHalluc",      0.63, 0.68, 0.73, 0.75, 0.83, 0.88],
        ["eSNLI-Halluc",   0.60, 0.68, 0.70, 0.72, 0.77, 0.84],
        ["HotPot-Halluc",  0.65, 0.64, 0.69, 0.72, 0.80, 0.87],
        ["Average",        0.63, 0.66, 0.71, 0.73, 0.80, 0.86],
    ],
    columns=[
        "dataset",
        "Baseline ALP",
        "ProbALP",
        "IG-Only",
        "Disc-Abduction",
        "IG-Abduction",
        "IG+Counter-Abduction",
    ],
)


IG_EFFICIENCY = pd.DataFrame(
    [
        ["Baseline ALP",            1.00, np.nan],
        ["ProbALP",                 1.35, np.nan],
        ["Disc-Abduction",          0.88, 12],
        ["IG-Abduction",            0.82, 18],
        ["IG+Counter-Abduction",    0.79, 21],
    ],
    columns=[
        "system",
        "solver_time_s",
        "search_space_reduction_pct",
    ],
)


IG_DEFEATED = pd.DataFrame(
    [
        ["Baseline ALP",          19],
        ["ProbALP",               15],
        ["Disc-Abduction",        13],
        ["IG-Abduction",           7],
        ["IG+Counter-Abduction",   6],
    ],
    columns=["system", "defeated_hypotheses_pct"],
)


IG_HUMAN = pd.DataFrame(
    [
        ["Baseline ALP",   3.1, 2.9, 2.8],
        ["ProbALP",        3.3, 3.0, 3.0],
        ["Disc-Abduction", 4.0, 3.8, 3.9],
        ["IG-Abduction",   4.4, 4.3, 4.2],
    ],
    columns=["system", "clarity", "coherence", "trust"],
)


IG_TRUST = pd.DataFrame(
    [
        ["Baseline ALP",          0.58, 0.65, 0.07],
        ["ProbALP",               0.57, 0.67, 0.10],
        ["IG+Counter-Abduction",  0.55, 0.78, 0.23],
    ],
    columns=["system", "trust_before", "trust_after", "delta_trust"],
)


# ================================================================
# 2. INPUT FORMAT
# ================================================================

"""
Each evaluated instance is stored as one JSON object.

Example:

{
    "instance_id": "med_001",
    "dataset": "Med-Halluc",
    "regime": "mixed",

    "gold_hallucination": 1,
    "pred_hallucination": 1,
    "pred_confidence": 0.94,

    "gold_escalate": 1,
    "pred_escalate": 1,

    "accepted": 0,
    "hard_constraint_violation": 0,

    "reasoning_hallucination": 1,
    "detected_hallucination": 1,

    "gold_failure_step": "s3",
    "pred_failure_step": "s3",

    "gold_failure_type": "bridge_error",
    "pred_failure_type": "bridge_error",

    "gold_failure_validator": "smt",
    "pred_failure_validator": "smt",

    "steps": [
        {
            "step_id": "s1",
            "logic": "alp",
            "gold_valid": 1,
            "pred_valid": 1
        },
        {
            "step_id": "s2",
            "logic": "smt",
            "gold_valid": 0,
            "pred_valid": 0
        }
    ],

    "bridges": [
        {
            "bridge_id": "b1",
            "from_logic": "alp",
            "to_logic": "smt",
            "gold_valid": 0,
            "pred_valid": 0
        }
    ],

    "chain_gold_certifiable": 0,
    "chain_pred_certified": 0,

    "latency": {
        "decomposition": 0.30,
        "routing": 0.05,
        "formalization": 0.25,
        "solver": 0.18,
        "certificate": 0.03,
        "bridge": 0.04,
        "counter_abduction": 0.20,
        "graph": 0.01,
        "total": 1.06
    }
}
"""


# ================================================================
# 3. I/O
# ================================================================

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Bad JSON at line {line_no} of {path}: {exc}"
                ) from exc

    return rows


def save_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
):

    with open(path, "w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(
                    dict(record),
                    ensure_ascii=False
                )
                + "\n"
            )


# ================================================================
# 4. BASIC METRICS
# ================================================================

def binary_metrics(
    gold: Sequence[int],
    pred: Sequence[int],
):

    if len(gold) == 0:

        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(gold, pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ================================================================
# 5. CALIBRATION
# ================================================================

def expected_calibration_error(
    gold,
    pred,
    confidence,
    n_bins=10,
):

    gold = np.asarray(gold)
    pred = np.asarray(pred)
    confidence = np.asarray(confidence, dtype=float)

    mask = ~np.isnan(confidence)

    gold = gold[mask]
    pred = pred[mask]
    confidence = confidence[mask]

    if len(gold) == 0:
        return np.nan

    correctness = (gold == pred).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0.0

    for lower, upper in zip(bins[:-1], bins[1:]):

        if upper == 1:

            selected = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            selected = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        if not np.any(selected):
            continue

        accuracy_bin = correctness[selected].mean()

        confidence_bin = confidence[selected].mean()

        ece += (
            selected.mean()
            *
            abs(
                accuracy_bin
                -
                confidence_bin
            )
        )

    return float(ece)


# ================================================================
# 6. STEP VALIDITY ACCURACY
# ================================================================

def step_validity_accuracy(records):

    gold = []
    pred = []

    for record in records:

        for step in record.get("steps", []):

            if (
                "gold_valid" in step
                and
                "pred_valid" in step
            ):

                gold.append(
                    int(step["gold_valid"])
                )

                pred.append(
                    int(step["pred_valid"])
                )

    if not gold:
        return np.nan

    return accuracy_score(gold, pred)


# ================================================================
# 7. BRIDGE VALIDATION
# ================================================================

def bridge_validation_accuracy(records):

    gold = []
    pred = []

    for record in records:

        for bridge in record.get("bridges", []):

            if (
                "gold_valid" in bridge
                and
                "pred_valid" in bridge
            ):

                gold.append(
                    int(bridge["gold_valid"])
                )

                pred.append(
                    int(bridge["pred_valid"])
                )

    if not gold:
        return np.nan

    return accuracy_score(gold, pred)


def bridge_false_accept_rate(records):

    errors = []

    for record in records:

        for bridge in record.get("bridges", []):

            if (
                int(bridge.get("gold_valid", 1)) == 0
            ):

                errors.append(
                    int(
                        bridge.get(
                            "pred_valid",
                            0
                        )
                    )
                )

    if not errors:
        return np.nan

    return np.mean(errors)


def bridge_false_reject_rate(records):

    errors = []

    for record in records:

        for bridge in record.get("bridges", []):

            if (
                int(
                    bridge.get(
                        "gold_valid",
                        0
                    )
                )
                ==
                1
            ):

                errors.append(
                    int(
                        bridge.get(
                            "pred_valid",
                            1
                        )
                        ==
                        0
                    )
                )

    if not errors:
        return np.nan

    return np.mean(errors)


# ================================================================
# 8. CHAIN CERTIFICATION
# ================================================================

def chain_certification_rate(records):
    """
    Fraction of reasoning chains certified by the system.
    """

    values = [
        int(r["chain_pred_certified"])
        for r in records
        if "chain_pred_certified" in r
    ]

    if not values:
        return np.nan

    return np.mean(values)


def chain_certification_accuracy(records):
    """
    Accuracy of the certification decision relative to gold.
    """

    gold = []
    pred = []

    for r in records:

        if (
            "chain_gold_certifiable" in r
            and
            "chain_pred_certified" in r
        ):

            gold.append(
                int(
                    r["chain_gold_certifiable"]
                )
            )

            pred.append(
                int(
                    r["chain_pred_certified"]
                )
            )

    if not gold:
        return np.nan

    return accuracy_score(gold, pred)


# ================================================================
# 9. FAILURE LOCALIZATION
# ================================================================

def failure_localization_accuracy(records):

    values = []

    for r in records:

        gold = r.get(
            "gold_failure_step"
        )

        pred = r.get(
            "pred_failure_step"
        )

        if gold is not None:

            values.append(
                int(
                    gold == pred
                )
            )

    if not values:
        return np.nan

    return np.mean(values)


def failure_type_accuracy(records):

    values = []

    for r in records:

        gold = r.get(
            "gold_failure_type"
        )

        pred = r.get(
            "pred_failure_type"
        )

        if gold is not None:

            values.append(
                int(
                    gold == pred
                )
            )

    if not values:
        return np.nan

    return np.mean(values)


def validator_localization_accuracy(records):

    values = []

    for r in records:

        gold = r.get(
            "gold_failure_validator"
        )

        pred = r.get(
            "pred_failure_validator"
        )

        if gold is not None:

            values.append(
                int(
                    gold == pred
                )
            )

    if not values:
        return np.nan

    return np.mean(values)


# ================================================================
# 10. ESCALATION
# ================================================================

def escalation_accuracy(records):

    gold = []
    pred = []

    for r in records:

        if (
            "gold_escalate" in r
            and
            "pred_escalate" in r
        ):

            gold.append(
                int(
                    r["gold_escalate"]
                )
            )

            pred.append(
                int(
                    r["pred_escalate"]
                )
            )

    if not gold:
        return np.nan

    return accuracy_score(gold, pred)


# ================================================================
# 11. HARD CONSTRAINT VIOLATIONS
# ================================================================

def constraint_violation_rate(records):

    accepted = [
        r
        for r in records
        if int(
            r.get(
                "accepted",
                0
            )
        )
        ==
        1
    ]

    if not accepted:
        return np.nan

    return np.mean(
        [
            int(
                r.get(
                    "hard_constraint_violation",
                    0
                )
            )
            for r in accepted
        ]
    )


# ================================================================
# 12. RRHR
# ================================================================

def relative_reasoning_hallucination_rate(records):

    detected = [
        r
        for r in records
        if int(
            r.get(
                "detected_hallucination",
                r.get(
                    "pred_hallucination",
                    0
                )
            )
        )
        ==
        1
    ]

    if not detected:
        return np.nan

    reasoning_failures = sum(

        int(
            r.get(
                "reasoning_hallucination",
                0
            )
        )

        for r in detected
    )

    return (
        reasoning_failures
        /
        len(detected)
    )


# ================================================================
# 13. FULL INSTANCE EVALUATION
# ================================================================

def evaluate_records(records):

    valid_records = [
        r
        for r in records
        if (
            "gold_hallucination" in r
            and
            "pred_hallucination" in r
        )
    ]

    gold = [
        int(
            r["gold_hallucination"]
        )
        for r in valid_records
    ]

    pred = [
        int(
            r["pred_hallucination"]
        )
        for r in valid_records
    ]

    confidence = [
        float(
            r.get(
                "pred_confidence",
                np.nan
            )
        )
        for r in valid_records
    ]

    metrics = binary_metrics(
        gold,
        pred
    )

    metrics.update(
        {

            "escalation":
                escalation_accuracy(
                    records
                ),

            "violation_rate":
                constraint_violation_rate(
                    records
                ),

            "calibration_error":
                expected_calibration_error(
                    gold,
                    pred,
                    confidence
                ),

            "sva":
                step_validity_accuracy(
                    records
                ),

            "ccr":
                chain_certification_rate(
                    records
                ),

            "cca":
                chain_certification_accuracy(
                    records
                ),

            "bva":
                bridge_validation_accuracy(
                    records
                ),

            "bridge_false_accept":
                bridge_false_accept_rate(
                    records
                ),

            "bridge_false_reject":
                bridge_false_reject_rate(
                    records
                ),

            "fla":
                failure_localization_accuracy(
                    records
                ),

            "failure_type_accuracy":
                failure_type_accuracy(
                    records
                ),

            "validator_localization":
                validator_localization_accuracy(
                    records
                ),

            "rrhr":
                relative_reasoning_hallucination_rate(
                    records
                ),

            "n":
                len(records),
        }
    )

    return metrics


# ================================================================
# 14. GROUPED EVALUATION
# ================================================================

def evaluate_by(
    records,
    field,
):

    groups = {}

    for r in records:

        value = str(
            r.get(
                field,
                "UNKNOWN"
            )
        )

        groups.setdefault(
            value,
            []
        ).append(r)

    rows = []

    for value, subset in groups.items():

        result = evaluate_records(
            subset
        )

        result[field] = value

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# ================================================================
# 15. MIXED-LOGIC ANALYSIS
# ================================================================

def evaluate_mixed_logic(records):

    mixed = [
        r
        for r in records
        if r.get("regime") == "mixed"
    ]

    result = evaluate_records(
        mixed
    )

    result.update(
        {
            "bridge_accuracy":
                bridge_validation_accuracy(
                    mixed
                ),

            "bridge_false_accept":
                bridge_false_accept_rate(
                    mixed
                ),

            "bridge_false_reject":
                bridge_false_reject_rate(
                    mixed
                ),
        }
    )

    return result


# ================================================================
# 16. LATENCY
# ================================================================

LATENCY_STAGES = [

    "decomposition",

    "routing",

    "formalization",

    "solver",

    "certificate",

    "bridge",

    "counter_abduction",

    "graph",

    "total",
]


def latency_summary(records):

    output = []

    for stage in LATENCY_STAGES:

        values = []

        for r in records:

            latency = r.get(
                "latency",
                {}
            )

            value = latency.get(
                stage
            )

            if value is not None:

                values.append(
                    float(value)
                )

        if values:

            output.append(
                {
                    "stage": stage,
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "p95": np.percentile(
                        values,
                        95
                    ),
                    "n": len(values),
                }
            )

        else:

            output.append(
                {
                    "stage": stage,
                    "mean": np.nan,
                    "median": np.nan,
                    "p95": np.nan,
                    "n": 0,
                }
            )

    return pd.DataFrame(
        output
    )


# ================================================================
# 17. ABLATION EXPERIMENTS
# ================================================================

def evaluate_run_directory(run_dir):

    run_dir = Path(
        run_dir
    )

    rows = []

    for path in sorted(
        run_dir.glob(
            "*.jsonl"
        )
    ):

        records = load_jsonl(
            path
        )

        result = evaluate_records(
            records
        )

        result["variant"] = path.stem

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# Expected files could be:

"""
runs/
    full_metapcr.jsonl
    no_provenance.jsonl
    no_temporal.jsonl
    no_defeasibility.jsonl
    no_certificates.jsonl
    no_bridges.jsonl
    no_counter_abduction.jsonl
    single_logic.jsonl
"""


# ================================================================
# 18. HUMAN EVALUATION
# ================================================================

"""
Expected CSV:

participant_id,
instance_id,
condition,
gold_hallucination,
human_pred,
confidence_before,
confidence_after,
interpretability,
failure_localization,
auditability
"""


def evaluate_human(path):

    df = pd.read_csv(
        path
    )

    results = []

    for condition, group in df.groupby(
        "condition"
    ):

        accuracy = accuracy_score(
            group["gold_hallucination"],
            group["human_pred"]
        )

        delta_confidence = np.nan

        if (
            "confidence_before" in group
            and
            "confidence_after" in group
        ):

            delta_confidence = (

                group[
                    "confidence_after"
                ]
                -
                group[
                    "confidence_before"
                ]

            ).mean()

        kappa = cohen_kappa_score(
            group["gold_hallucination"],
            group["human_pred"]
        )

        results.append(
            {

                "condition":
                    condition,

                "human_accuracy":
                    accuracy,

                "delta_confidence":
                    delta_confidence,

                "kappa_vs_gold":
                    kappa,

                "interpretability":
                    group[
                        "interpretability"
                    ].mean()
                    if
                    "interpretability"
                    in group
                    else np.nan,

                "failure_localization":
                    group[
                        "failure_localization"
                    ].mean()
                    if
                    "failure_localization"
                    in group
                    else np.nan,

                "auditability":
                    group[
                        "auditability"
                    ].mean()
                    if
                    "auditability"
                    in group
                    else np.nan,

                "n":
                    len(group),
            }
        )

    return pd.DataFrame(
        results
    )


# ================================================================
# 19. LATEX EXPORT
# ================================================================

def to_latex(
    df,
    caption,
    label,
):

    return df.to_latex(

        index=False,

        escape=False,

        na_rep="--",

        float_format=lambda x:
            f"{x:.3f}",

        caption=caption,

        label=label,

        position="H",
    )


# ================================================================
# 20. GENERATE COMPLETE REPORT
# ================================================================

def generate_report(
    prediction_file,
    output_dir,
    system_name="Full MetaPCR-LLM",
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    records = load_jsonl(
        prediction_file
    )

    # Overall
    overall = pd.DataFrame(
        [
            {
                "method":
                    system_name,

                **evaluate_records(
                    records
                )
            }
        ]
    )

    # Dataset results
    dataset_results = evaluate_by(
        records,
        "dataset"
    )

    # Reasoning regimes
    regime_results = evaluate_by(
        records,
        "regime"
    )

    # Mixed logic
    mixed = pd.DataFrame(
        [
            evaluate_mixed_logic(
                records
            )
        ]
    )

    # Latency
    latency = latency_summary(
        records
    )

    # Save CSV
    overall.to_csv(
        output_dir /
        "overall.csv",
        index=False
    )

    dataset_results.to_csv(
        output_dir /
        "dataset_results.csv",
        index=False
    )

    regime_results.to_csv(
        output_dir /
        "regime_results.csv",
        index=False
    )

    mixed.to_csv(
        output_dir /
        "mixed_logic.csv",
        index=False
    )

    latency.to_csv(
        output_dir /
        "latency.csv",
        index=False
    )

    # Save LaTeX
    (
        output_dir /
        "overall.tex"
    ).write_text(

        to_latex(
            overall,
            "Overall MetaPCR-LLM evaluation.",
            "tab:metapcr_overall"
        ),

        encoding="utf-8"
    )

    (
        output_dir /
        "datasets.tex"
    ).write_text(

        to_latex(
            dataset_results,
            "Dataset-level MetaPCR-LLM evaluation.",
            "tab:metapcr_dataset_results"
        ),

        encoding="utf-8"
    )

    (
        output_dir /
        "regimes.tex"
    ).write_text(

        to_latex(
            regime_results,
            "Performance by reasoning regime.",
            "tab:metapcr_regimes"
        ),

        encoding="utf-8"
    )

    (
        output_dir /
        "mixed_logic.tex"
    ).write_text(

        to_latex(
            mixed,
            "Performance on mixed-logic reasoning.",
            "tab:metapcr_mixed"
        ),

        encoding="utf-8"
    )

    (
        output_dir /
        "latency.tex"
    ).write_text(

        to_latex(
            latency,
            "End-to-end verifier latency.",
            "tab:metapcr_latency"
        ),

        encoding="utf-8"
    )

    print("\nOVERALL\n")

    print(
        overall.to_string(
            index=False
        )
    )

    print("\nBY DATASET\n")

    print(
        dataset_results.to_string(
            index=False
        )
    )

    print("\nBY REASONING REGIME\n")

    print(
        regime_results.to_string(
            index=False
        )
    )

    print("\nMIXED LOGIC\n")

    print(
        mixed.to_string(
            index=False
        )
    )

    print("\nLATENCY\n")

    print(
        latency.to_string(
            index=False
        )
    )


# ================================================================
# 21. EXPORT PUBLISHED REFERENCES
# ================================================================

def export_reference_tables(
    output_dir
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    tables = {

        "validllp_overall":
            VALIDLLP_OVERALL,

        "validllp_regime":
            VALIDLLP_REGIME,

        "validllp_dataset_f1":
            VALIDLLP_DATASET_F1,

        "validllp_ablation":
            VALIDLLP_ABLATION,

        "validllp_human":
            VALIDLLP_HUMAN,

        "validllp_rrhr":
            VALIDLLP_RRHR,

        "ig_dataset_f1":
            IG_DATASET_F1,

        "ig_efficiency":
            IG_EFFICIENCY,

        "ig_defeated":
            IG_DEFEATED,

        "ig_human":
            IG_HUMAN,

        "ig_trust":
            IG_TRUST,
    }

    for name, table in tables.items():

        table.to_csv(
            output_dir /
            f"{name}.csv",
            index=False
        )

        (
            output_dir /
            f"{name}.tex"
        ).write_text(

            table.to_latex(
                index=False,
                escape=False,
                na_rep="--",
                float_format=lambda x:
                    f"{x:.2f}"
            ),

            encoding="utf-8"
        )


# ================================================================
# 22. CLI
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True
    )

    # ------------------------------------------------------------

    evaluate_parser = sub.add_parser(
        "evaluate"
    )

    evaluate_parser.add_argument(
        "--predictions",
        required=True
    )

    evaluate_parser.add_argument(
        "--out",
        required=True
    )

    evaluate_parser.add_argument(
        "--name",
        default="Full MetaPCR-LLM"
    )

    # ------------------------------------------------------------

    run_parser = sub.add_parser(
        "evaluate-runs"
    )

    run_parser.add_argument(
        "--run-dir",
        required=True
    )

    run_parser.add_argument(
        "--out",
        required=True
    )

    # ------------------------------------------------------------

    human_parser = sub.add_parser(
        "human"
    )

    human_parser.add_argument(
        "--input",
        required=True
    )

    human_parser.add_argument(
        "--out",
        required=True
    )

    # ------------------------------------------------------------

    reference_parser = sub.add_parser(
        "export-reference"
    )

    reference_parser.add_argument(
        "--out",
        required=True
    )

    # ------------------------------------------------------------

    args = parser.parse_args()

    if args.command == "evaluate":

        generate_report(
            args.predictions,
            args.out,
            args.name
        )

    elif args.command == "evaluate-runs":

        result = evaluate_run_directory(
            args.run_dir
        )

        result.to_csv(
            args.out,
            index=False
        )

        print(
            result.to_string(
                index=False
            )
        )

    elif args.command == "human":

        result = evaluate_human(
            args.input
        )

        result.to_csv(
            args.out,
            index=False
        )

        print(
            result.to_string(
                index=False
            )
        )

    elif args.command == "export-reference":

        export_reference_tables(
            args.out
        )


if __name__ == "__main__":
    main()