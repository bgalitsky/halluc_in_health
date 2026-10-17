"""Inferential statistics (Sections 5.18 and 5.20).

* Source-clustered paired bootstrap for any metric difference (B = 10,000 by default).
* Two p-values are returned and kept apart, because they mean different things:
    - p_boot : empirical two-sided bootstrap p (shift method). Its resolution is
               1/(B+1); with B = 10,000 it can never be below ~1e-4.
    - p_norm : normal approximation using the bootstrap SE. Only this one can
               produce values like 1e-7, and the paper must say so if it is used.
* Holm step-down adjustment over an explicitly named family.
* Exact McNemar test, refused when units are clustered (repeated source_id).
* check_reported_p(): consistency check of a reported p-value against its CI
  (the check that flagged Table 20 in the review).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .schema import check_paired


# ---------------------------------------------------------------- bootstrap
#
# Resampling uses per-cluster sufficient statistics, so B = 10,000 resamples of
# thousands of instances run in well under a second. Any metric that is a ratio
# of sums (accuracy, F1, SVA, CCA, BVA, ProcessBench F1, ...) fits this form.

def _acc_suff(r):
    return [float(r["pred_halluc"] is not None and r["pred_halluc"] == r["gold_halluc"]), 1.0]

def _f1_suff(r):
    p = bool(r["pred_halluc"]) if r["pred_halluc"] is not None else False
    g = bool(r["gold_halluc"])
    return [float(p and g), float(p and not g), float((not p) and g)]

def _f1_fin(v):
    tp, fp, fn = v
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else np.nan

def _cca_suff(r):
    return [float(r["chain_pred_valid"] is not None and r["chain_pred_valid"] == r["chain_gold_valid"]), 1.0]

def _sva_suff(r):
    st = r.get("steps") or []
    return [float(sum(s["pred_valid"] is not None and s["pred_valid"] == s["gold_valid"] for s in st)), float(len(st))]

def _bva_suff(r):
    br = r.get("bridges") or []
    return [float(sum(b["pred_valid"] is not None and b["pred_valid"] == b["gold_valid"] for b in br)), float(len(br))]

def _ratio(v):
    return v[0] / v[1] if v[1] else np.nan

def _pb_suff(r):
    # ProcessBench: label = index of first erroneous step, -1 if the solution is correct
    ok = float(r["pred_first_error"] == r["gold_first_error"])
    err = r["gold_first_error"] != -1
    return [ok if err else 0.0, float(err), 0.0 if err else ok, float(not err)]

def _pb_fin(v):
    a_err = v[0] / v[1] if v[1] else np.nan
    a_ok = v[2] / v[3] if v[3] else np.nan
    return 2 * a_err * a_ok / (a_err + a_ok) if (a_err + a_ok) else 0.0

def _qacc_suff(r):
    return [float(r["pred_label"] == r["gold_label"]), 1.0]

SUFFICIENT = {
    "accuracy": (_acc_suff, _ratio),
    "f1": (_f1_suff, _f1_fin),
    "cca": (_cca_suff, _ratio),
    "sva": (_sva_suff, _ratio),
    "bva": (_bva_suff, _ratio),
    "processbench_f1": (_pb_suff, _pb_fin),
    "query_accuracy": (_qacc_suff, _ratio),
}


def _cluster_matrix(rows, suff, order):
    pos = {k: i for i, k in enumerate(order)}
    width = len(suff(rows[0]))
    M = np.zeros((len(order), width))
    for r in rows:
        M[pos[r["source_id"]]] += suff(r)
    return M


def paired_cluster_bootstrap(
    rows_a: list[dict],
    rows_b: list[dict],
    metric: str,
    B: int = 10_000,
    seed: int = 42081,
    alpha: float = 0.05,
    names: tuple[str, str] = ("A", "B"),
) -> dict:
    """Difference metric(A) - metric(B), resampling whole source clusters with
    replacement so all variants of a source stay together (Section 5.20).
    `metric` is a key of SUFFICIENT."""
    check_paired(rows_a, rows_b, *names)
    suff, fin = SUFFICIENT[metric]
    order = sorted({r["source_id"] for r in rows_a})
    Ma = _cluster_matrix(rows_a, suff, order)
    Mb = _cluster_matrix(rows_b, suff, order)
    obs = fin(Ma.sum(0)) - fin(Mb.sum(0))
    rng = np.random.default_rng(seed)
    n_cl = len(order)
    W = rng.multinomial(n_cl, np.full(n_cl, 1 / n_cl), size=B).astype(float)  # B x clusters
    Sa, Sb = W @ Ma, W @ Mb
    diffs = np.array([fin(Sa[i]) - fin(Sb[i]) for i in range(B)])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    se = float(np.std(diffs, ddof=1))
    exceed = int(np.sum(np.abs(diffs - obs) >= abs(obs)))   # shift method
    p_boot = (exceed + 1) / (len(diffs) + 1)
    p_norm = math.erfc(abs(obs) / (se * math.sqrt(2))) if se > 0 else 0.0
    return {
        "metric": metric, "diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "se": se,
        "p_boot": p_boot, "p_boot_floor": 1 / (len(diffs) + 1), "p_norm": p_norm,
        "B_effective": int(len(diffs)), "n_clusters": n_cl,
        "n_instances": len(rows_a), "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


# ---------------------------------------------------------------- Holm

def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values. Monotone in the raw ordering by construction."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for k, (name, p) in enumerate(items):
        adj = min(1.0, (m - k) * p)
        running = max(running, adj)
        out[name] = running
    return out


# ---------------------------------------------------------------- McNemar

def mcnemar_exact(rows_a, rows_b, names=("A", "B")) -> dict:
    """Exact two-sided McNemar on per-instance correctness. Valid only when each
    source contributes one instance; otherwise use the clustered bootstrap."""
    check_paired(rows_a, rows_b, *names)
    sources = [r["source_id"] for r in rows_a]
    if len(set(sources)) != len(sources):
        raise ValueError("McNemar requires independent paired units; source_id repeats. "
                         "Use paired_cluster_bootstrap instead (Section 5.20).")
    ib = {r["instance_id"]: r for r in rows_b}
    b = c = 0
    for r in rows_a:
        ca = r["pred_halluc"] == r["gold_halluc"]
        cb = ib[r["instance_id"]]["pred_halluc"] == r["gold_halluc"]
        b += ca and not cb
        c += cb and not ca
    n = b + c
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_exact": p}


# ---------------------------------------------------------------- audit check

def implied_p_from_ci(diff: float, lo: float, hi: float, level: float = 0.95) -> float:
    z_crit = {0.95: 1.959964, 0.99: 2.575829, 0.90: 1.644854}[level]
    se = (hi - lo) / (2 * z_crit)
    return math.erfc(abs(diff) / (se * math.sqrt(2)))


def check_reported_p(table: list[dict], family_size: Optional[int] = None, tol_orders: float = 1.0) -> list[dict]:
    """For each row {name, diff, ci_low, ci_high, reported_p}: compute the p implied
    by the CI (normal approx), its Holm bound, and flag disagreements larger than
    `tol_orders` orders of magnitude. Also checks Holm rank monotonicity."""
    m = family_size or len(table)
    raw = {r["name"]: implied_p_from_ci(r["diff"], r["ci_low"], r["ci_high"]) for r in table}
    adj = holm(raw) if m == len(table) else {k: min(1, v * m) for k, v in raw.items()}
    out = []
    for r in table:
        rp = max(r["reported_p"], 1e-300)
        imp = max(adj[r["name"]], 1e-300)
        gap = abs(math.log10(rp) - math.log10(imp))
        if r.get("upper_bound"):           # reported as "p < bound": consistent if implied is below it
            ok = imp <= rp * 10 ** tol_orders
        else:
            ok = gap <= tol_orders
        out.append({**r, "implied_raw_p": raw[r["name"]], "implied_holm_p": adj[r["name"]],
                    "log10_gap": gap, "consistent": ok})
    # rank check: Holm preserves the ordering of raw p-values
    by_raw = sorted(out, key=lambda x: x["implied_raw_p"])
    rep = [x["reported_p"] for x in by_raw]
    bound = [bool(x.get("upper_bound")) for x in by_raw]
    # a later (weaker) contrast may not report a smaller p than an earlier one; "< bound" entries are lenient
    monotone = all(rep[i] <= rep[i + 1] * (1 + 1e-9) or bound[i] for i in range(len(rep) - 1))
    for x in out:
        x["family_rank_monotone"] = monotone
    return out
