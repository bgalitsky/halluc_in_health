# MetaPCR-LLM evaluation scripts

Python code for every analysis in Section 5 of *MetaPCR-LLM: Universal Proof-Carrying Reasoning Across Heterogeneous Logics* (R1). The scripts score system outputs. They do not contain the MetaPCR verifier itself, the LLM calls or the native solvers. You run your systems, write one JSON line per (system, instance), and these scripts produce the tables, statistics and checks.

## Quick start

```bash
pip install numpy            # the only dependency (Python 3.10+)
python run_evaluation.py synth --data data/                 # example inputs (synthetic, NOT paper results)
python run_evaluation.py all --data data/ --out results/    # all tables + statistics -> results/report.md
python run_evaluation.py check-reported                     # audit the manuscript's reported p-values
python tests/test_eval.py                                   # unit tests
```

`--quick` shrinks the scalability grid (about 30 s for everything). The full Section 5.14 grid took about 12 minutes on one CPU core; every bound held and the growth rates matched Table 11. Each section also runs alone: `tables`, `stats`, `external`, `human`, `agreement`, `formalization`, `latency`, `errors`, `scalability`.

## What maps to what

| Paper | Script / function | Input file in `data/` |
|---|---|---|
| Tables 2–3 dataset sizes, regimes, leakage check | `diagnostics.dataset_statistics` | `dataset_instances.jsonl` |
| Table 4 Acc, F1, Esc, Viol, Coverage, Brier, SVA, CCA | `tables.table_main` | `primary_predictions.jsonl` |
| Table 5 F1 by regime, macro mean, worst regime | `tables.table_regime` | same |
| Table 6 mixed-logic F1, BVA, CCA, FLA, conflict detection | `tables.table_mixed` | same |
| Table 7 ablations | `tables.table_ablation` (systems named `ablation:-...`) | same |
| Table 8 F1 by dataset, macro mean, range | `tables.table_dataset` | same |
| Tables 9–10 localization, RRHR | `tables.table_localization`, `tables.table_rrhr` | same |
| Table 11 / Section 5.14 bridge scalability | `scalability.run_grid` (asserts every bound) | none (simulated) |
| Table 12 multi-LLM recomposition | `tables.table_recomposition` | `recomposition_predictions.jsonl` |
| Table 13 human study | `agreement.human_study`, `agreement.overlap_fleiss` | `human_study.jsonl` |
| Table 15 ProcessBench, RAGTruth, ProofWriter | `external.*` | `external_predictions.jsonl` |
| Table 16 mixed-case screening ledger | `diagnostics.screening_ledger` | `screening_ledger.jsonl` |
| Table 17 composition controls + per-origin breakdown | `tables.table_composition_controls` | `composition_predictions.jsonl` |
| Tables 18–19 latency | `diagnostics.latency_tables` | `primary_predictions.jsonl` (`latency`) |
| Table 20 / Section 5.20 statistics | `stats.paired_cluster_bootstrap`, `stats.holm` | predictions |
| Table 21 error analysis | `diagnostics.error_analysis` | `error_analysis.jsonl` |
| Table 22 checker acceptance and retries | `diagnostics.formalization_acceptance` | `formalization_attempts.jsonl` |
| Sections 5.4 and 5.18 agreement, audit fidelity | `agreement.agreement_report`, `diagnostics.formalization_fidelity` | `annotations_original.jsonl`, `audit_250.jsonl` |

Field definitions for every file are in `metapcr_eval/schema.py`. The synthetic files in `data/` are worked examples of each format.

## Running the external benchmarks on a real system

```python
from metapcr_eval import external as X
rows = X.load_processbench("processbench/gsm8k.json")             # official release
m = X.make_manifest("ProcessBench", "v1.0", "test", [r["instance_id"] for r in rows], n=450)
X.save_manifest(m, "manifests/processbench.json")                 # freeze BEFORE inference
keep = set(X.load_manifest("manifests/processbench.json")["ids"])
preds = X.run_verifier("processbench", my_verifier, [r for r in rows if r["instance_id"] in keep])
```

`my_verifier` implements the `Verifier` protocol in `external.py`. Gold fields are stripped before the verifier sees an instance. Failures are logged and mapped to a fixed fallback, so they are scored rather than dropped. Add `"task"` and append the output to `external_predictions.jsonl`.

## Design decisions tied to the reviewer comments

- **Clustered paired bootstrap.** Resamples whole `source_id` clusters, so variants of one source stay together. It uses per-cluster sufficient statistics, which makes B = 10,000 fast.
- **Two kinds of p-value, kept apart.**
  - `p_boot` is the empirical bootstrap p. It cannot go below 1/(B+1).
  - `p_norm` is a normal approximation from the bootstrap SE.

  Holm is applied over an explicitly named family. The paper must state which p it reports (Comment 1).
- **`check-reported`.** Recomputes the p implied by each reported CI and checks Holm rank order. On the R1 manuscript it flags all four Table 20 values. The original-study values are consistent with their CIs.
- **McNemar refuses clustered data.** It raises an error when `source_id` repeats.
- **Table 22 reports two retry means.** It gives the mean over all obligations and the mean over obligations that failed the first attempt. It also enforces the cap k (Comment 5).
- **Semantic fidelity is reported separately from checker acceptance.** It comes from the audit, by validator (Comment 3).
- **Agreement statistics.** Raw agreement and Fleiss' κ are computed from the raw labels before adjudication. Cohen's κ is added for two raters (Comment 2).
- **Composition controls are also scored per case origin.** This includes released, transformed, stress and local-pass cases (Comment 6).
- **Indeterminate predictions are never dropped.** They count against accuracy and recall.

## Limits

The synthetic generator only exercises the code; its numbers are not the paper's. Gold annotations, the systems and the native solvers must come from the authors' records. Adjudicated audit labels should replace the first-rater placeholder in `run_agreement`.
