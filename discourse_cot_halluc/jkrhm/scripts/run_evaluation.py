#!/usr/bin/env python3
"""Run repeated-split evaluation for the discourse hallucination detector."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else config_path.parent / p


def feature_columns(df: pd.DataFrame, include_groups: Iterable[str] | None = None) -> List[str]:
    excluded = {"id", "hallucination_label"}
    cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if include_groups is None:
        return cols
    prefixes = tuple(include_groups)
    return [c for c in cols if c.startswith(prefixes)]


def make_classifier(name: str, seed: int):
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= max_fpr)[0]
    return float(np.max(tpr[valid])) if valid.size else 0.0


def split_data(X, y, seed: int, train_size: float, validation_size: float, test_size: float):
    # First split test, then split remaining into train/validation.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )
    val_frac_of_trainval = validation_size / max(train_size + validation_size, 1e-9)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac_of_trainval, random_state=seed,
        stratify=y_trainval if len(np.unique(y_trainval)) > 1 else None
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_one(df: pd.DataFrame, cols: List[str], cfg: Dict[str, Any], seed: int, method_name: str) -> Dict[str, Any]:
    y = df["hallucination_label"].astype(int).to_numpy()
    X = df[cols].fillna(0.0).to_numpy(dtype=float)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, seed, cfg["model"]["train_size"], cfg["model"]["validation_size"], cfg["model"]["test_size"]
    )
    clf = make_classifier(cfg["model"].get("classifier", "logistic_regression"), seed)
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        test_score = clf.predict_proba(X_test)[:, 1]
    else:
        test_score = clf.decision_function(X_test)
    threshold = float(cfg["thresholds"].get("hallucination_probability", 0.5))
    y_pred = (test_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    try:
        auroc = roc_auc_score(y_test, test_score)
    except Exception:
        auroc = float("nan")
    metrics = {
        "method": method_name,
        "seed": seed,
        "n_features": len(cols),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
    }
    for fpr in cfg["thresholds"].get("tpr_at_fpr", [0.10, 0.05]):
        metrics[f"tpr_at_fpr_{int(fpr*100)}"] = tpr_at_fpr(y_test, test_score, float(fpr))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--method-name", default="full")
    parser.add_argument("--groups", nargs="*", default=None, help="Optional feature prefixes/groups, e.g. lex top disc ns jkrhm")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    features_path = resolve_path(config_path, args.features or cfg["outputs"]["features_csv"])
    output_path = resolve_path(config_path, args.output or cfg["outputs"]["metrics_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_path)
    groups = args.groups
    cols = feature_columns(df, groups)
    if not cols:
        raise SystemExit("No feature columns selected.")
    rows = [evaluate_one(df, cols, cfg, int(seed), args.method_name) for seed in cfg["model"]["random_seeds"]]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote repeated-split metrics to {output_path}")


if __name__ == "__main__":
    main()
