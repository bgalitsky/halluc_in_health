"""Annotation agreement (Sections 5.4 and 5.18) and the human study (Section 5.16)."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Optional

import numpy as np


# ---------------------------------------------------------------- agreement

def raw_agreement(labels: list[list[str]]) -> float:
    """Mean pairwise agreement per item; labels[i] = ratings of item i (>= 2 raters)."""
    vals = []
    for item in labels:
        n = len(item)
        c = Counter(item)
        vals.append(sum(v * (v - 1) for v in c.values()) / (n * (n - 1)))
    return mean(vals)


def fleiss_kappa(labels: list[list[str]]) -> float:
    """Fleiss' kappa; every item must have the same number of raters."""
    n = len(labels[0])
    if any(len(x) != n for x in labels):
        raise ValueError("Fleiss' kappa needs a constant number of raters per item")
    cats = sorted({c for item in labels for c in item})
    N = len(labels)
    M = np.array([[Counter(item)[c] for c in cats] for item in labels], dtype=float)
    P_i = (np.sum(M * M, axis=1) - n) / (n * (n - 1))
    p_j = M.sum(0) / (N * n)
    P_bar, P_e = P_i.mean(), np.sum(p_j ** 2)
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for exactly two raters (the conventional choice for the 250-case audit)."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement_report(rows: list[dict], field: str) -> dict:
    """rows: {item_id, annotator, <field>} BEFORE adjudication."""
    items = defaultdict(list)
    for r in rows:
        items[r["item_id"]].append(r[field])
    labels = list(items.values())
    rep = {"field": field, "n_items": len(labels), "n_raters": len(labels[0]),
           "raw_agreement": raw_agreement(labels), "fleiss_kappa": fleiss_kappa(labels)}
    if rep["n_raters"] == 2:
        rep["cohen_kappa"] = cohen_kappa([x[0] for x in labels], [x[1] for x in labels])
    return rep


# ---------------------------------------------------------------- human study

def normalize_likert(x: float) -> float:
    return (x - 1) / 4


def human_study(rows: list[dict], B: int = 5000, seed: int = 7) -> list[dict]:
    """rows: {evaluator, item_id, condition, correct: bool, confidence_0_100,
    interpretability_1_5, failure_loc_correct: bool|null}. 'LLM only' is the
    unaided baseline seen first; every other condition is the post-report rating.

    Reported descriptively (Section 5.20): judgments are clustered by evaluator
    and item, and report viewing follows the unaided assessment, so the CI below
    resamples evaluators (cluster bootstrap) and is not an order-controlled
    causal estimate."""
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    base = {(r["evaluator"], r["item_id"]): r for r in by_cond.get("LLM only", [])}
    evaluators = sorted({r["evaluator"] for r in rows})
    rng = np.random.default_rng(seed)
    out = []
    for cond, rs in by_cond.items():
        acc = mean(float(r["correct"]) for r in rs)
        interp = mean(normalize_likert(r["interpretability_1_5"]) for r in rs)
        fl = [r["failure_loc_correct"] for r in rs if r.get("failure_loc_correct") is not None]
        dconf = None
        if cond != "LLM only":
            d = [r["confidence_0_100"] / 100 - base[(r["evaluator"], r["item_id"])]["confidence_0_100"] / 100
                 for r in rs if (r["evaluator"], r["item_id"]) in base]
            dconf = mean(d) if d else None
        # evaluator-cluster bootstrap CI of accuracy
        per_e = defaultdict(list)
        for r in rs:
            per_e[r["evaluator"]].append(float(r["correct"]))
        sums = np.array([sum(per_e[e]) for e in evaluators])
        cnts = np.array([len(per_e[e]) for e in evaluators])
        W = rng.multinomial(len(evaluators), np.full(len(evaluators), 1 / len(evaluators)), size=B)
        boots = (W @ sums) / np.maximum(W @ cnts, 1)
        out.append({"condition": cond, "human_acc": acc,
                    "acc_ci_evaluator_cluster": tuple(np.quantile(boots, [0.025, 0.975])),
                    "delta_conf": dconf, "interpretability_norm": interp,
                    "interpretability_likert": 1 + 4 * interp,
                    "failure_loc": mean(fl) if fl else None, "n_judgments": len(rs)})
    order = ["LLM only", "+ Logic Program", "+ Single-logic PCR", "+ ValidLLP-style report", "+ Full MetaPCR-LLM"]
    out.sort(key=lambda x: order.index(x["condition"]) if x["condition"] in order else 99)
    return out


def overlap_fleiss(rows: list[dict], condition: str) -> Optional[float]:
    """Fleiss' kappa on the overlap pool: items rated by the SAME number of evaluators."""
    items = defaultdict(list)
    for r in rows:
        if r["condition"] == condition:
            items[r["item_id"]].append(str(r["judgment"]))
    sizes = Counter(len(v) for v in items.values() if len(v) >= 2)
    if not sizes:
        return None
    k = sizes.most_common(1)[0][0]
    pool = [v for v in items.values() if len(v) == k]
    return fleiss_kappa(pool)


def latency_summary(values: list[float]) -> dict:
    v = sorted(values)
    p95 = float(np.quantile(v, 0.95))
    m = mean(v)
    return {"mean": m, "median": median(v), "p95": p95,
            "warning": ("mean exceeds P95: check for extreme outliers or a reporting error" if m > p95 else None)}
