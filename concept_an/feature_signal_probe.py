#!/usr/bin/env python3
"""
feature_signal_probe.py

Demonstrates that Concept Axis Accuracy (CAP) on the discourse-CW dataset is
bottlenecked by the FEATURES, not by the model or the loss function.

The script measures the *achievable* concept separability -- an upper bound on
the CAP any aligned model could reach -- using a single, fixed, standard
classifier (logistic regression). It varies only ONE thing: how much each
concept's 8 features differ from the other concepts'.

    signal strength = 0.0  -> the original data (features were sampled per-label,
                              so concepts overlap within a label)
    signal strength > 0.0  -> a fixed per-concept "signature" is added to the
                              features, so concepts become progressively
                              distinguishable in feature space

Because the classifier is held fixed across the whole sweep and only the data
changes, a rising separability shows that CAP is limited by the information in
the features -- not by the training objective.

Caveats to state honestly when reporting this:
  * The logistic-regression probe is a stand-in for "best achievable CAP" (the
    ceiling), not the concept-whitening model itself.
  * The injected signatures are random per-concept directions: a synthetic
    stand-in for the genuine concept-discriminative discourse features that real
    data would supply. The point is the RELATIONSHIP between feature signal and
    achievable CAP, not the specific signatures.

Usage:
  python feature_signal_probe.py --data data/discourse_cw_hallucination_dataset_1000.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "unsupported_nucleus",
    "evidence_density",
    "contrast_density",
    "closure_strength",
    "weak_to_strong_inference",
    "support_balance",
    "defeater_presence",
    "nucleus_pressure",
]


def build_signatures(concepts, n_features, seed):
    """One fixed random unit-vector 'signature' per concept (reproducible)."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=(len(concepts), n_features))
    sig /= np.linalg.norm(sig, axis=1, keepdims=True)
    return {c: sig[i] for i, c in enumerate(concepts)}


def separability(X, y, cv, seed):
    """k-way concept separability via a fixed logistic-regression probe."""
    clf = LogisticRegression(max_iter=3000, C=10, random_state=seed)
    return cross_val_score(clf, X, y, cv=cv).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/discourse_cw_hallucination_dataset_1000.csv")
    p.add_argument("--strengths", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
                   help="signal strengths to sweep (0.0 = original data)")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    missing = [c for c in FEATURE_COLUMNS + ["discourse_subtype"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # standardized 8 features, and the concept label for each row
    X0 = StandardScaler().fit_transform(df[FEATURE_COLUMNS].astype(float).values)
    y = df["discourse_subtype"].astype(str).values
    concepts = sorted(np.unique(y))

    # one fixed signature per concept; add it (scaled) to each row's features
    sigs = build_signatures(concepts, X0.shape[1], args.seed)
    S = np.stack([sigs[c] for c in y])

    print(f"Loaded {len(df)} rows | {len(concepts)} concepts | {X0.shape[1]} features")
    print(f"Probe: logistic regression, {args.cv}-fold CV | chance = {1.0/len(concepts):.3f}")
    print()
    print(f"{'signal_strength':>16s}{'achievable_CAP':>18s}")
    print("-" * 34)
    for a in args.strengths:
        Xa = X0 + a * S
        acc = separability(Xa, y, args.cv, args.seed)
        tag = "   <- original data" if a == 0.0 else ""
        print(f"{a:>16.2f}{acc:>18.3f}{tag}")
    print()
    print("The classifier never changes across rows above; only how much the")
    print("features differ by concept changes. CAP rising with signal => CAP is")
    print("limited by the features, not by the model or the loss function.")


if __name__ == "__main__":
    main()
