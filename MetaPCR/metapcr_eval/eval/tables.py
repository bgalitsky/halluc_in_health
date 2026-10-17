"""Builds every results table of Section 5 from per-instance predictions."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

from . import metrics as M
from .schema import by_system

SINGLE_REGIMES = ("rule", "uncertainty", "abduction", "conflict", "constraint")
FULL = "Full MetaPCR-LLM"
VALIDLLP = "ValidLLP-style"
NOBRIDGE = "MetaPCR w/o bridges"


# ---------------------------------------------------------------- rendering

def to_markdown(rows: list[dict], cols: Optional[list[str]] = None, nd: int = 2, title: str = "") -> str:
    if not rows:
        return f"### {title}\n\n(no data)\n"
    cols = cols or list(rows[0])
    out = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                if v != 0 and abs(v) < 1e-3:
                    cells.append(f"{v:.2e}")
                else:
                    cells.append(M.fmt(v, 3 if abs(v) < 0.01 and v != 0 else nd))
            elif isinstance(v, tuple):
                cells.append("[" + ", ".join(M.fmt(x, 3) for x in v) + "]")
            elif v is None:
                cells.append("–")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def to_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _ordered(systems: Iterable[str], order: Optional[list[str]]) -> list[str]:
    s = list(systems)
    if not order:
        return sorted(s)
    return [x for x in order if x in s] + sorted(x for x in s if x not in order)


# ---------------------------------------------------------------- Table 4

def table_main(rows, order=None):
    out = []
    for s, rs in ((k, by_system(rows)[k]) for k in _ordered(by_system(rows), order)):
        if s.startswith("ablation:"):
            continue
        out.append({"Method": s, "Acc.": M.accuracy(rs), "F1": M.f1(rs), "Esc.": M.escalation_appropriateness(rs),
                    "Viol.↓": M.violation_rate(rs), "Coverage↑": M.coverage(rs), "Brier↓": M.brier(rs),
                    "SVA": M.step_validity_accuracy(rs), "CCA": M.chain_certification_accuracy(rs)})
    return out


# ---------------------------------------------------------------- Table 5

def table_regime(rows, order=None):
    out = []
    bs = by_system(rows)
    for s in _ordered(bs, order):
        if s.startswith("ablation:"):
            continue
        rec = {"Method": s}
        for g in SINGLE_REGIMES:
            rec[g] = M.f1([r for r in bs[s] if r["regime"] == g])
        rec["Mean"] = M.macro_mean(rec[g] for g in SINGLE_REGIMES)
        rec["Min"] = min((rec[g] for g in SINGLE_REGIMES if rec[g] is not None), default=None)
        out.append(rec)
    return out


# ---------------------------------------------------------------- Table 6

def table_mixed(rows, order=None):
    out = []
    bs = by_system(rows)
    for s in _ordered(bs, order):
        if s.startswith("ablation:"):
            continue
        rs = [r for r in bs[s] if r["regime"] == "mixed"]
        out.append({"Method": s, "F1": M.f1(rs), "Bridge Acc.": M.bridge_validation_accuracy(rs),
                    "CCA": M.chain_certification_accuracy(rs), "Failure Loc.": M.failure_localization_accuracy(rs),
                    "Conflict Detection": M.conflict_detection(rs), "n": len(rs)})
    return out


# ---------------------------------------------------------------- Table 7

def table_ablation(rows, full=FULL):
    bs = by_system(rows)
    names = [full] + sorted(s for s in bs if s.startswith("ablation:"))
    base_f1 = M.f1(bs[full])
    out = []
    for s in names:
        rs = bs[s]
        f = M.f1(rs)
        out.append({"Variant": s.replace("ablation:-", "– ").replace("ablation:", "– ") if s != full else s, "Accuracy": M.accuracy(rs),
                    "Hallucination F1": f, "CCA": M.chain_certification_accuracy(rs),
                    "FLA": M.failure_localization_accuracy(rs), "Escalation": M.escalation_appropriateness(rs),
                    "ΔF1 vs full": None if s == full else base_f1 - f})
    return out


# ---------------------------------------------------------------- Table 8

def table_dataset(rows, order=None):
    bs = by_system(rows)
    datasets = sorted({r["dataset"] for r in rows})
    out = []
    for s in _ordered(bs, order):
        if s.startswith("ablation:"):
            continue
        rec = {"Method": s}
        for d in datasets:
            rec[d] = M.f1([r for r in bs[s] if r["dataset"] == d])
        vals = [rec[d] for d in datasets if rec[d] is not None]
        rec["Mean"] = M.macro_mean(vals)
        rec["Range"] = max(vals) - min(vals) if vals else None
        out.append(rec)
    return out


# ---------------------------------------------------------------- Tables 9–10

def table_localization(rows, order=None):
    bs = by_system(rows)
    return [{"Method": s, "Step Localization": M.failure_localization_accuracy(bs[s]),
             "Failure Type": M.failure_type_accuracy(bs[s]), "Correct Validator": M.correct_validator_accuracy(bs[s]),
             "Bridge Failure": M.bridge_failure_localization(bs[s])}
            for s in _ordered(bs, order) if not s.startswith("ablation:")]


def table_rrhr(rows, order=None):
    bs = by_system(rows)
    return [{"Method": s, "RRHR↓": M.rrhr(bs[s])} for s in _ordered(bs, order) if not s.startswith("ablation:")]


# ---------------------------------------------------------------- Table 12

def table_recomposition(rows, order=None):
    bs = by_system(rows)
    return [{"Strategy": s, "Answer Acc.": M.accuracy(bs[s]), "F1": M.f1(bs[s]),
             "CCA": M.chain_certification_accuracy(bs[s]), "Invalid Compositions↓": M.invalid_composition_rate(bs[s])}
            for s in _ordered(bs, order)]


# ---------------------------------------------------------------- Table 17

def table_composition_controls(rows, order=None):
    """Pooled overview AND the per-origin breakdown the review asked for."""
    bs = by_system(rows)
    pooled, per_origin = [], []
    for s in _ordered(bs, order):
        rs = bs[s]
        pooled.append({"Method": s, "Hallucination F1": M.f1(rs), "CCA": M.chain_certification_accuracy(rs),
                       "BVA": M.bridge_validation_accuracy(rs), "Indet.": M.indeterminate_fraction(rs),
                       "n_unique": len({r["instance_id"] for r in rs})})
        for o in sorted({r.get("origin") for r in rs}, key=str):
            ro = [r for r in rs if r.get("origin") == o]
            per_origin.append({"Method": s, "origin": o, "n": len(ro), "F1": M.f1(ro),
                               "CCA": M.chain_certification_accuracy(ro), "BVA": M.bridge_validation_accuracy(ro)})
    return pooled, per_origin
