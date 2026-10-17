"""Formalization acceptance (5.23), latency (5.19), error analysis (5.21),
mixed-case screening ledger (5.18, Table 16), and dataset statistics (5.2–5.3)."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Optional

from .agreement import latency_summary


# ---------------------------------------------------------------- Table 22

def formalization_acceptance(attempt_log: list[dict], cap: int = 5) -> list[dict]:
    """attempt_log rows: {validator, obligation_id, attempts: [bool, ...]} where
    attempts[0] is the first try and each later entry is one refinement retry.

    Reports BOTH mean retries over all obligations and over obligations that
    failed the first attempt, so the column cannot be misread (the review found
    a 3.2 mean with 88.3% first-pass acceptance, which is only possible conditionally).
    These are checker-acceptance rates, NOT semantic formalization accuracy."""
    by_v = defaultdict(list)
    for r in attempt_log:
        retries = len(r["attempts"]) - 1
        if retries > cap:
            raise ValueError(f"{r['obligation_id']}: {retries} retries exceeds cap k={cap}")
        by_v[r["validator"]].append(r)
    out = []
    for v, rs in sorted(by_v.items()):
        retries = [len(r["attempts"]) - 1 for r in rs]
        failed_first = [len(r["attempts"]) - 1 for r in rs if not r["attempts"][0]]
        out.append({"validator": v, "n_obligations": len(rs),
                    "first_acceptance": mean(r["attempts"][0] for r in rs),
                    "final_acceptance": mean(any(r["attempts"]) for r in rs),
                    "mean_retries_all": mean(retries),
                    "mean_retries_given_first_failure": mean(failed_first) if failed_first else 0.0,
                    "n_failed_first": len(failed_first), "cap": cap})
    tot = [r for rs in by_v.values() for r in rs]
    out.append({"validator": "POOLED (instance-weighted)", "n_obligations": len(tot),
                "first_acceptance": mean(r["attempts"][0] for r in tot),
                "final_acceptance": mean(any(r["attempts"]) for r in tot),
                "mean_retries_all": mean(len(r["attempts"]) - 1 for r in tot),
                "mean_retries_given_first_failure": mean([len(r["attempts"]) - 1 for r in tot if not r["attempts"][0]] or [0]),
                "n_failed_first": sum(not r["attempts"][0] for r in tot), "cap": cap})
    return out


def formalization_fidelity(audit_rows: list[dict]) -> list[dict]:
    """Semantic fidelity from the independent audit (adjudicated labels):
    rows {validator, faithful: bool}. This is what Comment 3 actually asked for."""
    by_v = defaultdict(list)
    for r in audit_rows:
        by_v[r["validator"]].append(bool(r["faithful"]))
    return [{"validator": v, "n_audited": len(x), "fidelity_accuracy": mean(x)} for v, x in sorted(by_v.items())]


# ---------------------------------------------------------------- Tables 18–19

STAGES = ("decomp", "native", "bridges", "counter_abd", "aggregation")


def latency_tables(rows_by_system: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    stage, wall = [], []
    for sysname, rows in rows_by_system.items():
        lat = [r["latency"] for r in rows if r.get("latency")]
        if not lat:
            continue
        means = {s: (mean(l[s] for l in lat if l.get(s) is not None) if any(l.get(s) is not None for l in lat) else None)
                 for s in STAGES}
        seq = sum(v for v in means.values() if v is not None)
        stage.append({"system": sysname, **means, "sequential_sum": seq,
                      **{f"share_{s}": (means[s] / seq if means[s] and seq else None) for s in STAGES}})
        wall.append({"system": sysname, **latency_summary([l["wall"] for l in lat])})
    return stage, wall


# ---------------------------------------------------------------- Table 21

ERROR_CATEGORIES = ("Semantic decomposition", "Logic routing", "Formalization",
                    "Cross-logic bridge", "Native-validator limitation")


def error_analysis(rows: list[dict]) -> list[dict]:
    """rows: {instance_id, category} for the manually inspected stratified sample."""
    c = Counter(r["category"] for r in rows)
    unknown = set(c) - set(ERROR_CATEGORIES)
    if unknown:
        raise ValueError(f"unknown error categories: {unknown}")
    n = sum(c.values())
    out = [{"category": k, "count": c[k], "pct": 100 * c[k] / n} for k in ERROR_CATEGORIES]
    out.append({"category": "Routing + bridge", "count": c["Logic routing"] + c["Cross-logic bridge"],
                "pct": 100 * (c["Logic routing"] + c["Cross-logic bridge"]) / n})
    out.append({"category": "Decomposition + formalization",
                "count": c["Semantic decomposition"] + c["Formalization"],
                "pct": 100 * (c["Semantic decomposition"] + c["Formalization"]) / n})
    out.append({"category": "Total", "count": n, "pct": 100.0})
    return out


# ---------------------------------------------------------------- Table 16

def screening_ledger(ledger: list[dict]) -> list[dict]:
    """ledger rows: {case_id, origin, disposition in {mixed, not_mixed, ambiguous, excluded}}.
    The four dispositions must partition the examined cases."""
    by_o = defaultdict(Counter)
    for r in ledger:
        by_o[r["origin"]][r["disposition"]] += 1
    out = []
    for o, c in sorted(by_o.items()):
        ex = sum(c.values())
        out.append({"origin": o, "examined": ex, "mixed": c["mixed"], "not_mixed": c["not_mixed"],
                    "ambiguous": c["ambiguous"], "excluded": c["excluded"], "mixed_rate": c["mixed"] / ex})
    return out


# ---------------------------------------------------------------- Tables 2–3

def dataset_statistics(instances: list[dict]) -> tuple[list[dict], list[dict]]:
    """instances: {instance_id, source_id, dataset, partition, gold_halluc, regime} (one row per instance)."""
    by_d = defaultdict(list)
    for r in instances:
        by_d[r["dataset"]].append(r)
    # leakage check: one source -> one partition
    part_of = defaultdict(set)
    for r in instances:
        part_of[(r["dataset"], r["source_id"])].add(r["partition"])
    leaks = [k for k, v in part_of.items() if len(v) > 1]
    if leaks:
        raise ValueError(f"{len(leaks)} sources appear in more than one partition, e.g. {leaks[:3]}")
    t2, t3 = [], []
    regimes = ("rule", "uncertainty", "abduction", "conflict", "constraint", "mixed")
    for d, rs in sorted(by_d.items()):
        p = Counter(r["partition"] for r in rs)
        t2.append({"dataset": d, "train": p["train"], "validation": p["validation"], "test": p["test"],
                   "total": len(rs), "halluc_rate": mean(r["gold_halluc"] for r in rs)})
        rg = Counter(r["regime"] for r in rs)
        t3.append({"dataset": d, **{g: rg[g] / len(rs) for g in regimes}})
    allr = list(instances)
    t2.append({"dataset": "Total", "train": sum(x["train"] for x in t2), "validation": sum(x["validation"] for x in t2),
               "test": sum(x["test"] for x in t2), "total": len(allr), "halluc_rate": mean(r["gold_halluc"] for r in allr)})
    return t2, t3
