"""Metric definitions from Section 5.6 and the later result sections.

All functions take a list of InstanceRecord dicts for ONE system and return a float
(or None when the metric is undefined for that system, rendered as "–" in tables).
"""
from __future__ import annotations

import math
from statistics import mean
from typing import Optional


def _safe_div(n: float, d: float) -> Optional[float]:
    return None if d == 0 else n / d


# ---------- answer level ----------------------------------------------------

def accuracy(rows) -> Optional[float]:
    """Fraction of final decisions matching gold. Indeterminate (None) counts as wrong."""
    return _safe_div(sum(r["pred_halluc"] is not None and r["pred_halluc"] == r["gold_halluc"] for r in rows), len(rows))


def confusion(rows, pred_key="pred_halluc", gold_key="gold_halluc"):
    """Hallucination = positive class. Indeterminate predictions count as negative
    (not flagged), i.e. they can only cost recall — they are never silently removed."""
    tp = fp = fn = tn = 0
    for r in rows:
        p = bool(r[pred_key]) if r[pred_key] is not None else False
        g = bool(r[gold_key])
        tp += p and g
        fp += p and not g
        fn += (not p) and g
        tn += (not p) and (not g)
    return tp, fp, fn, tn


def f1(rows, pred_key="pred_halluc", gold_key="gold_halluc") -> Optional[float]:
    tp, fp, fn, _ = confusion(rows, pred_key, gold_key)
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    if not p or not r:
        return 0.0 if (p is not None and r is not None) else None
    return 2 * p * r / (p + r)


def escalation_appropriateness(rows) -> Optional[float]:
    """Esc = N_correct escalation decisions / N_instances."""
    return _safe_div(sum(r["pred_escalate"] == r["gold_escalate"] for r in rows), len(rows))


def coverage(rows) -> Optional[float]:
    """Coverage = N_accepted / N_all."""
    return _safe_div(sum(bool(r["accepted"]) for r in rows), len(rows))


def violation_rate(rows) -> Optional[float]:
    """Viol = N_accepted violating a hard constraint / N_accepted. Report with coverage."""
    acc = [r for r in rows if r["accepted"]]
    return _safe_div(sum(bool(r["violates_hard_constraint"]) for r in acc), len(acc))


def brier(rows) -> Optional[float]:
    """BS = mean (p_i - y_i)^2 with y = gold hallucination."""
    vals = [(float(r["p_halluc"]) - float(r["gold_halluc"])) ** 2 for r in rows if r.get("p_halluc") is not None]
    return mean(vals) if vals else None


def indeterminate_fraction(rows) -> Optional[float]:
    return _safe_div(sum(r["pred_halluc"] is None for r in rows), len(rows))


# ---------- proof level -----------------------------------------------------

def step_validity_accuracy(rows) -> Optional[float]:
    """SVA over the SHARED gold decomposition. A step with no prediction (None) is wrong."""
    n = c = 0
    for r in rows:
        for s in r.get("steps") or []:
            n += 1
            c += s["pred_valid"] is not None and s["pred_valid"] == s["gold_valid"]
    return _safe_div(c, n)


def chain_certification_accuracy(rows) -> Optional[float]:
    return _safe_div(
        sum(r["chain_pred_valid"] is not None and r["chain_pred_valid"] == r["chain_gold_valid"] for r in rows),
        len(rows),
    )


def bridge_validation_accuracy(rows) -> Optional[float]:
    """BVA; None ('–') for systems without explicit bridge objects."""
    if all(r.get("bridges") is None for r in rows):
        return None
    n = c = 0
    for r in rows:
        for b in r.get("bridges") or []:
            n += 1
            c += b["pred_valid"] is not None and b["pred_valid"] == b["gold_valid"]
    return _safe_div(c, n)


def failure_localization_accuracy(rows) -> Optional[float]:
    """FLA = cases where the gold defective step is localized / cases with a gold defect."""
    d = [r for r in rows if r.get("gold_defect_step") is not None]
    return _safe_div(sum(r.get("pred_defect_step") == r["gold_defect_step"] for r in d), len(d))


def _match_rate(rows, gold_key, pred_key):
    d = [r for r in rows if r.get(gold_key) is not None]
    if not d or all(r.get(pred_key) is None for r in d):
        return None
    return sum(r.get(pred_key) == r[gold_key] for r in d) / len(d)


def failure_type_accuracy(rows):
    return _match_rate(rows, "gold_failure_type", "pred_failure_type")


def correct_validator_accuracy(rows):
    return _match_rate(rows, "gold_validator", "pred_validator")


def bridge_failure_localization(rows):
    return _match_rate(rows, "gold_bridge_failure", "pred_bridge_failure")


def rrhr(rows) -> Optional[float]:
    """Residual Reasoning Hallucination Rate: gold-invalid chains that were accepted."""
    bad = [r for r in rows if r["chain_gold_valid"] is False]
    return _safe_div(sum(r["chain_pred_valid"] is True for r in bad), len(bad))


def conflict_detection(rows) -> Optional[float]:
    d = [r for r in rows if r.get("gold_conflict") is not None]
    return _safe_div(sum(r.get("pred_conflict") == r["gold_conflict"] for r in d), len(d))


# ---------- multi-LLM recomposition ----------------------------------------

def invalid_composition_rate(rows) -> Optional[float]:
    """Invalid compositions / attempted compositions, pooled over instances."""
    tot = sum(r.get("n_compositions", 0) for r in rows)
    bad = sum(r.get("n_invalid_compositions", 0) for r in rows)
    return _safe_div(bad, tot)


METRICS = {
    "Acc.": accuracy,
    "F1": f1,
    "Esc.": escalation_appropriateness,
    "Viol.": violation_rate,
    "Coverage": coverage,
    "Brier": brier,
    "SVA": step_validity_accuracy,
    "CCA": chain_certification_accuracy,
    "BVA": bridge_validation_accuracy,
    "FLA": failure_localization_accuracy,
    "RRHR": rrhr,
    "Indet.": indeterminate_fraction,
}


def macro_mean(values) -> Optional[float]:
    """Unweighted arithmetic mean (the 'Mean' columns); need not equal pooled F1."""
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def fmt(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"
