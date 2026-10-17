#!/usr/bin/env python3
"""MetaPCR-LLM evaluation driver: reproduces every table and test in Section 5.

  python run_evaluation.py synth --data data/            # write synthetic example inputs
  python run_evaluation.py all   --data data/ --out results/ [--quick]
  python run_evaluation.py check-reported                # audit the paper's Table 20 p-values

Each section can also be run alone: tables | stats | external | human | agreement |
formalization | latency | errors | scalability.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval import agreement as A
from eval import diagnostics as D
from eval import external as X
from eval import scalability as SC
from eval import stats as ST
from eval import tables as T
from eval.schema import by_system, read_jsonl

ORDER = ["LP", "LP + IC", "PLP", "ALP", "Argumentation/DeLP", "SMT/CLP", "Single-logic PCR",
         "ValidLLP-style", "MetaPCR w/o bridges", "Full MetaPCR-LLM"]
EXT_ORDER = ["LLM-based verifier", "ValidLLP-style", "SAFE", "TP-as-a-Judge", "VeriCoT",
             "MetaPCR w/o bridge checks", "Full MetaPCR-LLM"]


class Report:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.parts: list[str] = []

    def add(self, key: str, title: str, rows, cols=None, note: str = ""):
        T.to_csv(rows, self.out / f"{key}.csv")
        self.parts.append(T.to_markdown(rows, cols, title=title) + (f"\n{note}\n" if note else ""))

    def text(self, s: str):
        self.parts.append(s + "\n")

    def save(self):
        p = self.out / "report.md"
        p.write_text("# MetaPCR-LLM evaluation report\n\n" + "\n".join(self.parts))
        return p


def _load(data: Path, name: str):
    p = data / name
    return read_jsonl(p) if p.exists() else None


# ------------------------------------------------------------------ sections

def run_tables(data, rep):
    rows = _load(data, "primary_predictions.jsonl")
    if rows is None:
        return
    inst = _load(data, "dataset_instances.jsonl")
    if inst:
        t2, t3 = D.dataset_statistics(inst)
        rep.add("table02_datasets", "Table 2. Dataset statistics", t2)
        rep.add("table03_regimes", "Table 3. Distribution of reasoning regimes", t3)
    test = [r for r in rows]
    rep.add("table04_main", "Table 4. Overall validation performance (pooled F1)", T.table_main(test, ORDER))
    rep.add("table05_regime", "Table 5. F1 by dominant regime (Mean = unweighted macro)", T.table_regime(test, ORDER))
    rep.add("table06_mixed", "Table 6. Mixed-logic cases", T.table_mixed(test, ORDER))
    rep.add("table07_ablation", "Table 7. Ablations (one component removed at a time)", T.table_ablation(test),
            note="Single-component ablations show conditional contributions; deltas are not additive.")
    rep.add("table08_dataset", "Table 8. F1 by dataset", T.table_dataset(test, ORDER))
    rep.add("table09_localization", "Table 9. Failure localization", T.table_localization(test, ORDER))
    rep.add("table10_rrhr", "Table 10. Residual Reasoning Hallucination Rate", T.table_rrhr(test, ORDER))
    rec = _load(data, "recomposition_predictions.jsonl")
    if rec:
        rep.add("table12_recomposition", "Table 12. Multi-LLM recomposition (K = 3)", T.table_recomposition(rec, list(dict.fromkeys(r["system"] for r in rec))))
    comp = _load(data, "composition_predictions.jsonl")
    if comp:
        pooled, per_origin = T.table_composition_controls(comp)
        rep.add("table17_composition", "Table 17. Composition controls (pooled)", pooled)
        rep.add("table17b_composition_by_origin", "Table 17b. Composition controls by case origin", per_origin,
                note="Per-origin scores answer whether the bridge gain holds on unmodified released cases.")


def run_stats(data, rep, B):
    rows = _load(data, "primary_predictions.jsonl")
    out = {}
    if rows:
        bs = by_system(rows)
        mixed = {s: [r for r in v if r["regime"] == "mixed"] for s, v in bs.items()}
        contrasts = {
            "primary accuracy: Full vs ValidLLP": (bs[T.FULL], bs[T.VALIDLLP], "accuracy"),
            "primary pooled F1: Full vs ValidLLP": (bs[T.FULL], bs[T.VALIDLLP], "f1"),
            "mixed F1: Full vs ValidLLP": (mixed[T.FULL], mixed[T.VALIDLLP], "f1"),
            "mixed F1: Full vs no-bridge": (mixed[T.FULL], mixed[T.NOBRIDGE], "f1"),
        }
        res = {k: ST.paired_cluster_bootstrap(a, b, m, B=B, names=(k, "ref")) for k, (a, b, m) in contrasts.items()}
        adj = ST.holm({k: v["p_norm"] for k, v in res.items()})
        out["original"] = [{"contrast": k, **v, "holm_p_norm (family = these 4)": adj[k]} for k, v in res.items()]
    ext = _load(data, "external_predictions.jsonl")
    comp = _load(data, "composition_predictions.jsonl")
    fam = {}
    if ext:
        for task, (_, _, suff) in X.SCORERS.items():
            tr = by_system([r for r in ext if r["task"] == task])
            fam[f"{task}: Full vs ValidLLP"] = ST.paired_cluster_bootstrap(tr["Full MetaPCR-LLM"], tr["ValidLLP-style"], suff, B=B, names=("Full", "ValidLLP"))
    if comp:
        cb = by_system(comp)
        fam["pooled mixed: Full vs no-bridge"] = ST.paired_cluster_bootstrap(cb["Full MetaPCR-LLM"], cb["MetaPCR without bridge checks"], "f1", B=B, names=("Full", "nobridge"))
    if fam:
        adj = ST.holm({k: v["p_norm"] for k, v in fam.items()})
        out["extension"] = [{"contrast": k, **v, "holm_p_norm (family = these 4)": adj[k]} for k, v in fam.items()]
    cols = ["contrast", "diff", "ci_low", "ci_high", "n_clusters", "p_boot", "p_boot_floor", "p_norm",
            "holm_p_norm (family = these 4)", "ci_excludes_zero"]
    for k, title in (("original", "Statistics — original-study contrasts (Section 5.20)"),
                     ("extension", "Table 20. Statistics — extension contrasts")):
        if k in out:
            rep.add(f"stats_{k}", title, out[k], cols,
                    note=(f"B = {B} source-clustered paired bootstrap resamples. p_boot cannot go below p_boot_floor; "
                          "p_norm is a normal approximation from the bootstrap SE. Holm is applied to p_norm over the "
                          "named family of four contrasts. Report which p is used."))
    (rep.out / "stats.json").write_text(json.dumps(out, indent=1, default=float))


def run_external(data, rep):
    ext = _load(data, "external_predictions.jsonl")
    if not ext:
        return
    table = {}
    for task, (scorer, key, _) in X.SCORERS.items():
        for s, rs in by_system([r for r in ext if r["task"] == task]).items():
            table.setdefault(s, {"Method": s})[task] = scorer(rs)[key]
            fails = sum(r.get("status", "ok") != "ok" for r in rs)
            table[s][f"{task}_failures_scored"] = fails
    rows = [table[s] for s in EXT_ORDER if s in table] + [v for k, v in table.items() if k not in EXT_ORDER]
    rep.add("table15_external", "Table 15. External benchmarks (original annotations; no cross-task average)", rows,
            ["Method", "processbench", "ragtruth", "proofwriter"],
            note="ProcessBench: official F1 (harmonic mean of erroneous/correct accuracy). RAGTruth: response-level F1. "
                 "ProofWriter: open-world three-way query accuracy.")


def run_human(data, rep):
    h = _load(data, "human_study.jsonl")
    if not h:
        return
    rows = A.human_study(h)
    for r in rows:
        r["fleiss_overlap"] = A.overlap_fleiss(h, r["condition"])
    rep.add("table13_human", "Table 13. Human evaluation (descriptive)", rows,
            ["condition", "human_acc", "acc_ci_evaluator_cluster", "delta_conf", "fleiss_overlap",
             "interpretability_norm", "interpretability_likert", "failure_loc", "n_judgments"],
            note="Judgments are clustered and report viewing follows the unaided rating; not an order-controlled causal effect. "
                 "ΔConf is confidence change, not calibration.")


def run_agreement(data, rep):
    rows = []
    o = _load(data, "annotations_original.jsonl")
    if o:
        rows += [{"study": "original (6 annotators)", **A.agreement_report(o, f)} for f in ("regime", "bridge_required")]
    a = _load(data, "audit_250.jsonl")
    if a:
        rows += [{"study": "independent audit (2 experts)", **A.agreement_report(a, f)} for f in ("regime", "faithful")]
        adjudicated = {}
        for r in a:
            adjudicated.setdefault(r["item_id"], r)  # replace with the adjudicated label in real data
        rep.add("formalization_fidelity", "Semantic formalization fidelity by validator (from the audit)",
                D.formalization_fidelity(list(adjudicated.values())),
                note="This is measured fidelity to the source text, distinct from checker acceptance (Table 22).")
    if rows:
        rep.add("agreement", "Annotation agreement (before adjudication)", rows)


def run_formalization(data, rep):
    log = _load(data, "formalization_attempts.jsonl")
    if log:
        rep.add("table22_formalization", "Table 22. Native-checker acceptance and refinement retries (cap k = 5)",
                D.formalization_acceptance(log, cap=5),
                note="Acceptance by the checker does not establish fidelity to the source statement.")


def run_latency(data, rep):
    rows = _load(data, "primary_predictions.jsonl")
    if not rows:
        return
    bs = {s: v for s, v in by_system(rows).items() if s in ("Single-logic PCR", "ValidLLP-style", "MetaPCR w/o bridges", "Full MetaPCR-LLM")}
    stage, wall = D.latency_tables(bs)
    rep.add("table18_stage_latency", "Table 18. Mean stage latency (s) and sequential sum", stage,
            ["system", "decomp", "native", "bridges", "counter_abd", "aggregation", "sequential_sum"])
    rep.add("table19_wall_latency", "Table 19. End-to-end wall-clock latency (s)", wall)


def run_errors(data, rep):
    e = _load(data, "error_analysis.jsonl")
    if e:
        rep.add("table21_errors", "Table 21. Manually analyzed failures", D.error_analysis(e))
    l = _load(data, "screening_ledger.jsonl")
    if l:
        rep.add("table16_screening", "Table 16. Mixed-case screening ledger", D.screening_ledger(l))


def run_scalability(rep, quick):
    if quick:
        rows = SC.run_grid(ns=(2, 6, 10), Ls=(5, 20, 80), qs=(0, 0.5, 1.0), Ks=(1, 3, 10), graphs=20)
    else:
        rows = SC.run_grid()
    T.to_csv(rows, rep.out / "scalability_grid.csv")
    summ = []
    for strat, K in (("independent", 3), ("unpruned", 10), ("beam", 10)):
        for y in ("candidates_per_graph", "cross_logic_transitions_per_graph"):
            summ.append({"strategy": strat, "varied": "L", "held fixed": f"K={K}, q=1, n=10", "quantity": y,
                         "log-log slope": SC.growth_exponent(rows, "L", y, strategy=strat, q=1.0, n=10, K=K)})
    for strat in ("independent", "unpruned", "beam"):
        y = "candidates_per_graph" if strat == "independent" else "candidates_per_reached_transition"
        summ.append({"strategy": strat, "varied": "K", "held fixed": "L=80, q=1, n=10", "quantity": y,
                     "log-log slope": SC.growth_exponent(rows, "K", y, strategy=strat, q=1.0, n=10, L=80)})
    summ.append({"strategy": "independent", "varied": "q", "held fixed": "K=1, L=80, n=10", "quantity": "cross_logic_transitions_per_graph",
                 "log-log slope": SC.growth_exponent(rows, "q", "cross_logic_transitions_per_graph", strategy="independent", n=10, L=80, K=1)})
    rep.add("scalability_summary", "Section 5.14. Empirical growth exponents (log-log slopes)", summ,
            note="All Table 11 upper bounds are asserted on every configuration; full grid in scalability_grid.csv. "
                 "Expected: ~1 in L; ~1 in K for independent chains; ~2 in K unpruned; ~0 in K with beam width w = 3; "
                 "~1 in q. Recomposition stops when invalid bridges empty the frontier, so per-graph totals for "
                 "recomposition can grow sublinearly in L; per-reached-transition counts isolate the K dependence.")


# ------------------------------------------------------------------ audit of reported values

PAPER_TABLE20 = [
    {"name": "ProcessBench", "diff": 0.04, "ci_low": 0.018, "ci_high": 0.062, "reported_p": 1.06e-7},
    {"name": "RAGTruth", "diff": 0.04, "ci_low": 0.021, "ci_high": 0.059, "reported_p": 1e-16, "upper_bound": True},
    {"name": "ProofWriter", "diff": 0.13, "ci_low": 0.094, "ci_high": 0.166, "reported_p": 2.82e-7},
    {"name": "Pooled mixed", "diff": 0.11, "ci_low": 0.078, "ci_high": 0.142, "reported_p": 4.12e-9},
]
PAPER_ORIGINAL = [
    {"name": "Primary accuracy", "diff": 0.03, "ci_low": 0.0192, "ci_high": 0.0408, "reported_p": 1.06e-7},
    {"name": "Primary pooled F1", "diff": 0.06, "ci_low": 0.0481, "ci_high": 0.0719, "reported_p": 1e-16, "upper_bound": True},
    {"name": "Mixed F1 vs ValidLLP", "diff": 0.06, "ci_low": 0.0371, "ci_high": 0.0831, "reported_p": 2.82e-7},
    {"name": "Mixed F1 vs no-bridge", "diff": 0.10, "ci_low": 0.0767, "ci_high": 0.1240, "reported_p": 3.33e-16},
]


def check_reported(rep):
    for key, title, tab in (("check_table20", "Consistency of reported Table 20 p-values with their CIs", PAPER_TABLE20),
                            ("check_original", "Consistency of reported original-study p-values with their CIs", PAPER_ORIGINAL)):
        rows = ST.check_reported_p(tab)
        rep.add(key, title, rows, ["name", "diff", "ci_low", "ci_high", "reported_p", "implied_raw_p",
                                   "implied_holm_p", "log10_gap", "consistent", "family_rank_monotone"],
                note="Implied p uses a normal approximation from the CI width; '<1e-16' is entered as 1e-16. "
                     "A gap above 1 order of magnitude, or a non-monotone Holm ordering, needs explanation.")


# ------------------------------------------------------------------ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["synth", "all", "tables", "stats", "external", "human", "agreement",
                                        "formalization", "latency", "errors", "scalability", "check-reported"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--B", type=int, default=10_000, help="bootstrap resamples")
    ap.add_argument("--quick", action="store_true", help="reduced scalability grid")
    a = ap.parse_args(argv)
    data = Path(a.data)
    if a.command == "synth":
        from eval.synth import generate
        generate(data)
        print(f"synthetic inputs written to {data}/ (NOT paper results)")
        return 0
    rep = Report(Path(a.out))
    cmd = a.command
    if cmd in ("all", "tables"): run_tables(data, rep)
    if cmd in ("all", "external"): run_external(data, rep)
    if cmd in ("all", "stats"): run_stats(data, rep, a.B)
    if cmd in ("all", "human"): run_human(data, rep)
    if cmd in ("all", "agreement"): run_agreement(data, rep)
    if cmd in ("all", "formalization"): run_formalization(data, rep)
    if cmd in ("all", "latency"): run_latency(data, rep)
    if cmd in ("all", "errors"): run_errors(data, rep)
    if cmd in ("all", "scalability"): run_scalability(rep, a.quick)
    if cmd in ("all", "check-reported"): check_reported(rep)
    print(f"report: {rep.save()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
