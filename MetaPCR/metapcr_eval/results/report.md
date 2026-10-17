# MetaPCR-LLM evaluation report

### Table 2. Dataset statistics

| dataset | train | validation | test | total | halluc_rate |
|---|---|---|---|---|---|
| Autoimmune-narrate-halluc | 240 | 360 | 600 | 1200 | 0.38 |
| Med-Halluc | 400 | 600 | 1000 | 2000 | 0.54 |
| Truthful-Halluc | 200 | 300 | 500 | 1000 | 0.45 |
| eSNLI-Halluc | 200 | 300 | 500 | 1000 | 0.39 |
| Total | 1040 | 1560 | 2600 | 5200 | 0.45 |

### Table 3. Distribution of reasoning regimes

| dataset | rule | uncertainty | abduction | conflict | constraint | mixed |
|---|---|---|---|---|---|---|
| Autoimmune-narrate-halluc | 0.16 | 0.11 | 0.11 | 0.13 | 0.19 | 0.31 |
| Med-Halluc | 0.19 | 0.18 | 0.08 | 0.10 | 0.13 | 0.31 |
| Truthful-Halluc | 0.20 | 0.16 | 0.07 | 0.17 | 0.10 | 0.30 |
| eSNLI-Halluc | 0.15 | 0.18 | 0.07 | 0.09 | 0.15 | 0.35 |

### Table 4. Overall validation performance (pooled F1)

| Method | Acc. | F1 | Esc. | Viol.↓ | Coverage↑ | Brier↓ | SVA | CCA |
|---|---|---|---|---|---|---|---|---|
| LP | 0.69 | 0.66 | 0.70 | 0.01 | 0.53 | 0.25 | 0.70 | 0.67 |
| LP + IC | 0.77 | 0.75 | 0.71 | 0.01 | 0.53 | 0.20 | 0.74 | 0.70 |
| PLP | 0.87 | 0.85 | 0.70 | 0.004 | 0.54 | 0.13 | 0.79 | 0.82 |
| ALP | 0.84 | 0.83 | 0.70 | 0.01 | 0.55 | 0.14 | 0.76 | 0.79 |
| Argumentation/DeLP | 0.87 | 0.85 | 0.70 | 0.01 | 0.55 | 0.13 | 0.76 | 0.81 |
| SMT/CLP | 0.84 | 0.82 | 0.69 | 0.008 | 0.54 | 0.15 | 0.80 | 0.79 |
| Single-logic PCR | 0.79 | 0.77 | 0.68 | 0.01 | 0.53 | 0.19 | 0.82 | 0.72 |
| ValidLLP-style | 0.92 | 0.91 | 0.69 | 0.01 | 0.55 | 0.09 | 0.84 | 0.86 |
| MetaPCR w/o bridges | 0.90 | 0.88 | 0.71 | 0.008 | 0.54 | 0.11 | 0.84 | 0.82 |
| Full MetaPCR-LLM | 0.94 | 0.93 | 0.78 | 0.005 | 0.55 | 0.07 | 0.88 | 0.87 |

### Table 5. F1 by dominant regime (Mean = unweighted macro)

| Method | rule | uncertainty | abduction | conflict | constraint | Mean | Min |
|---|---|---|---|---|---|---|---|
| LP | 0.67 | 0.58 | 0.70 | 0.65 | 0.68 | 0.66 | 0.58 |
| LP + IC | 0.74 | 0.66 | 0.82 | 0.78 | 0.76 | 0.75 | 0.66 |
| PLP | 0.84 | 0.95 | 0.85 | 0.78 | 0.85 | 0.85 | 0.78 |
| ALP | 0.81 | 0.82 | 0.97 | 0.73 | 0.85 | 0.84 | 0.73 |
| Argumentation/DeLP | 0.86 | 0.78 | 0.88 | 0.98 | 0.81 | 0.86 | 0.78 |
| SMT/CLP | 0.79 | 0.79 | 0.81 | 0.83 | 0.95 | 0.84 | 0.79 |
| Single-logic PCR | 0.77 | 0.78 | 0.81 | 0.79 | 0.77 | 0.79 | 0.77 |
| ValidLLP-style | 0.92 | 0.93 | 0.92 | 0.91 | 0.92 | 0.92 | 0.91 |
| MetaPCR w/o bridges | 0.92 | 0.91 | 0.92 | 0.94 | 0.92 | 0.92 | 0.91 |
| Full MetaPCR-LLM | 0.93 | 0.94 | 0.96 | 0.95 | 0.91 | 0.94 | 0.91 |

### Table 6. Mixed-logic cases

| Method | F1 | Bridge Acc. | CCA | Failure Loc. | Conflict Detection | n |
|---|---|---|---|---|---|---|
| LP | 0.69 | – | 0.69 | 0.68 | 0.65 | 817 |
| LP + IC | 0.76 | – | 0.72 | 0.63 | 0.68 | 817 |
| PLP | 0.84 | – | 0.80 | 0.62 | 0.68 | 817 |
| ALP | 0.82 | – | 0.79 | 0.70 | 0.67 | 817 |
| Argumentation/DeLP | 0.84 | – | 0.81 | 0.67 | 0.66 | 817 |
| SMT/CLP | 0.79 | – | 0.78 | 0.70 | 0.65 | 817 |
| Single-logic PCR | 0.72 | – | 0.68 | 0.75 | 0.63 | 817 |
| ValidLLP-style | 0.87 | – | 0.84 | 0.79 | 0.65 | 817 |
| MetaPCR w/o bridges | 0.81 | – | 0.76 | 0.75 | 0.65 | 817 |
| Full MetaPCR-LLM | 0.93 | 0.89 | 0.84 | 0.80 | 0.86 | 817 |

### Table 7. Ablations (one component removed at a time)

| Variant | Accuracy | Hallucination F1 | CCA | FLA | Escalation | ΔF1 vs full |
|---|---|---|---|---|---|---|
| Full MetaPCR-LLM | 0.94 | 0.93 | 0.87 | 0.81 | 0.78 | – |
| – bridge_validation | 0.92 | 0.90 | 0.83 | 0.80 | 0.70 | 0.03 |
| – counter_abduction | 0.81 | 0.79 | 0.74 | 0.81 | 0.77 | 0.15 |
| – defeasibility_labels | 0.63 | 0.60 | 0.62 | 0.80 | 0.79 | 0.33 |
| – native_certificates | 0.57 | 0.53 | 0.57 | 0.80 | 0.79 | 0.40 |
| – provenance_labels | 0.81 | 0.79 | 0.74 | 0.81 | 0.77 | 0.14 |
| – temporal_labels | 0.68 | 0.65 | 0.66 | 0.78 | 0.78 | 0.29 |
| – uncertainty_labels | 0.74 | 0.72 | 0.70 | 0.79 | 0.78 | 0.21 |
| – single_target_logic | 0.67 | 0.64 | 0.64 | 0.81 | 0.78 | 0.29 |

Single-component ablations show conditional contributions; deltas are not additive.

### Table 8. F1 by dataset

| Method | Autoimmune-narrate-halluc | Med-Halluc | Truthful-Halluc | eSNLI-Halluc | Mean | Range |
|---|---|---|---|---|---|---|
| LP | 0.59 | 0.70 | 0.69 | 0.64 | 0.65 | 0.11 |
| LP + IC | 0.68 | 0.78 | 0.77 | 0.72 | 0.74 | 0.10 |
| PLP | 0.81 | 0.87 | 0.87 | 0.82 | 0.84 | 0.06 |
| ALP | 0.77 | 0.85 | 0.86 | 0.80 | 0.82 | 0.09 |
| Argumentation/DeLP | 0.81 | 0.88 | 0.85 | 0.82 | 0.84 | 0.07 |
| SMT/CLP | 0.78 | 0.86 | 0.79 | 0.80 | 0.81 | 0.07 |
| Single-logic PCR | 0.73 | 0.78 | 0.78 | 0.75 | 0.76 | 0.06 |
| ValidLLP-style | 0.89 | 0.94 | 0.90 | 0.87 | 0.90 | 0.07 |
| MetaPCR w/o bridges | 0.88 | 0.89 | 0.89 | 0.87 | 0.88 | 0.03 |
| Full MetaPCR-LLM | 0.92 | 0.94 | 0.94 | 0.93 | 0.93 | 0.03 |

### Table 9. Failure localization

| Method | Step Localization | Failure Type | Correct Validator | Bridge Failure |
|---|---|---|---|---|
| LP | 0.64 | 0.60 | 0.55 | – |
| LP + IC | 0.64 | 0.59 | 0.52 | – |
| PLP | 0.66 | 0.63 | 0.56 | – |
| ALP | 0.69 | 0.62 | 0.57 | – |
| Argumentation/DeLP | 0.69 | 0.68 | 0.63 | – |
| SMT/CLP | 0.68 | 0.66 | 0.60 | – |
| Single-logic PCR | 0.72 | 0.68 | 0.63 | – |
| ValidLLP-style | 0.78 | 0.74 | 0.70 | – |
| MetaPCR w/o bridges | 0.74 | 0.71 | 0.65 | – |
| Full MetaPCR-LLM | 0.81 | 0.75 | 0.72 | 0.69 |

### Table 10. Residual Reasoning Hallucination Rate

| Method | RRHR↓ |
|---|---|
| LP | 0.33 |
| LP + IC | 0.29 |
| PLP | 0.17 |
| ALP | 0.22 |
| Argumentation/DeLP | 0.19 |
| SMT/CLP | 0.22 |
| Single-logic PCR | 0.29 |
| ValidLLP-style | 0.13 |
| MetaPCR w/o bridges | 0.19 |
| Full MetaPCR-LLM | 0.14 |

### Table 12. Multi-LLM recomposition (K = 3)

| Strategy | Answer Acc. | F1 | CCA | Invalid Compositions↓ |
|---|---|---|---|---|
| Best single model | 0.83 | 0.81 | 0.76 | 0.29 |
| Whole-answer voting | 0.85 | 0.83 | 0.76 | 0.27 |
| Step recomposition | 0.88 | 0.86 | 0.81 | 0.23 |
| + native certificates | 0.94 | 0.93 | 0.85 | 0.20 |
| + certificates + bridges + labels | 0.94 | 0.94 | 0.86 | 0.18 |

### Table 17. Composition controls (pooled)

| Method | Hallucination F1 | CCA | BVA | Indet. | n_unique |
|---|---|---|---|---|---|
| Full MetaPCR-LLM | 0.93 | 0.86 | 0.87 | 0.02 | 1480 |
| MetaPCR without bridge checks | 0.80 | 0.75 | – | 0.03 | 1480 |
| Native-validator portfolio | 0.75 | 0.73 | – | 0.09 | 1480 |
| ValidLLP-style | 0.84 | 0.78 | – | 0.05 | 1480 |

### Table 17b. Composition controls by case origin

| Method | origin | n | F1 | CCA | BVA |
|---|---|---|---|---|---|
| Full MetaPCR-LLM | local_pass | 150 | 0.92 | 0.89 | 0.89 |
| Full MetaPCR-LLM | released | 420 | 0.93 | 0.87 | 0.88 |
| Full MetaPCR-LLM | stress | 400 | 0.92 | 0.86 | 0.86 |
| Full MetaPCR-LLM | transformed | 510 | 0.94 | 0.85 | 0.87 |
| MetaPCR without bridge checks | local_pass | 150 | 0.81 | 0.76 | – |
| MetaPCR without bridge checks | released | 420 | 0.82 | 0.74 | – |
| MetaPCR without bridge checks | stress | 400 | 0.81 | 0.75 | – |
| MetaPCR without bridge checks | transformed | 510 | 0.79 | 0.75 | – |
| Native-validator portfolio | local_pass | 150 | 0.75 | 0.75 | – |
| Native-validator portfolio | released | 420 | 0.78 | 0.74 | – |
| Native-validator portfolio | stress | 400 | 0.74 | 0.68 | – |
| Native-validator portfolio | transformed | 510 | 0.74 | 0.75 | – |
| ValidLLP-style | local_pass | 150 | 0.83 | 0.76 | – |
| ValidLLP-style | released | 420 | 0.83 | 0.79 | – |
| ValidLLP-style | stress | 400 | 0.85 | 0.79 | – |
| ValidLLP-style | transformed | 510 | 0.83 | 0.78 | – |

Per-origin scores answer whether the bridge gain holds on unmodified released cases.

### Table 15. External benchmarks (original annotations; no cross-task average)

| Method | processbench | ragtruth | proofwriter |
|---|---|---|---|
| LLM-based verifier | 0.61 | 0.64 | 0.71 |
| ValidLLP-style | 0.79 | 0.90 | 0.78 |
| SAFE | 0.73 | – | – |
| TP-as-a-Judge | 0.68 | – | – |
| VeriCoT | – | 0.85 | 0.87 |
| MetaPCR w/o bridge checks | 0.79 | 0.88 | 0.92 |
| Full MetaPCR-LLM | 0.85 | 0.93 | 0.94 |

ProcessBench: official F1 (harmonic mean of erroneous/correct accuracy). RAGTruth: response-level F1. ProofWriter: open-world three-way query accuracy.

### Statistics — original-study contrasts (Section 5.20)

| contrast | diff | ci_low | ci_high | n_clusters | p_boot | p_boot_floor | p_norm | holm_p_norm (family = these 4) | ci_excludes_zero |
|---|---|---|---|---|---|---|---|---|---|
| primary accuracy: Full vs ValidLLP | 0.02 | 0.01 | 0.03 | 1318 | 2.00e-04 | 1.00e-04 | 9.11e-06 | 1.82e-05 | True |
| primary pooled F1: Full vs ValidLLP | 0.03 | 0.01 | 0.04 | 1318 | 2.00e-04 | 1.00e-04 | 1.12e-05 | 1.82e-05 | True |
| mixed F1: Full vs ValidLLP | 0.06 | 0.03 | 0.09 | 653 | 1.00e-04 | 1.00e-04 | 3.44e-06 | 1.03e-05 | True |
| mixed F1: Full vs no-bridge | 0.12 | 0.10 | 0.15 | 653 | 1.00e-04 | 1.00e-04 | 1.39e-18 | 5.55e-18 | True |

B = 10000 source-clustered paired bootstrap resamples. p_boot cannot go below p_boot_floor; p_norm is a normal approximation from the bootstrap SE. Holm is applied to p_norm over the named family of four contrasts. Report which p is used.

### Table 20. Statistics — extension contrasts

| contrast | diff | ci_low | ci_high | n_clusters | p_boot | p_boot_floor | p_norm | holm_p_norm (family = these 4) | ci_excludes_zero |
|---|---|---|---|---|---|---|---|---|---|
| processbench: Full vs ValidLLP | 0.05 | 0.02 | 0.09 | 450 | 0.003 | 1.00e-04 | 0.003 | 0.006 | True |
| ragtruth: Full vs ValidLLP | 0.03 | 0.008 | 0.06 | 245 | 0.01 | 1.00e-04 | 0.01 | 0.01 | True |
| proofwriter: Full vs ValidLLP | 0.16 | 0.13 | 0.19 | 204 | 1.00e-04 | 1.00e-04 | 1.01e-20 | 3.02e-20 | True |
| pooled mixed: Full vs no-bridge | 0.12 | 0.11 | 0.14 | 740 | 1.00e-04 | 1.00e-04 | 9.63e-38 | 3.85e-37 | True |

B = 10000 source-clustered paired bootstrap resamples. p_boot cannot go below p_boot_floor; p_norm is a normal approximation from the bootstrap SE. Holm is applied to p_norm over the named family of four contrasts. Report which p is used.

### Table 13. Human evaluation (descriptive)

| condition | human_acc | acc_ci_evaluator_cluster | delta_conf | fleiss_overlap | interpretability_norm | interpretability_likert | failure_loc | n_judgments |
|---|---|---|---|---|---|---|---|---|
| LLM only | 0.65 | [0.597, 0.700] | – | 0.02 | 0.63 | 3.53 | 0.56 | 360 |
| + Logic Program | 0.76 | [0.717, 0.800] | 0.08 | -0.21 | 0.67 | 3.69 | 0.63 | 360 |
| + Single-logic PCR | 0.81 | [0.786, 0.836] | 0.12 | -0.02 | 0.72 | 3.90 | 0.71 | 360 |
| + ValidLLP-style report | 0.83 | [0.803, 0.858] | 0.17 | 0.17 | 0.78 | 4.11 | 0.76 | 360 |
| + Full MetaPCR-LLM | 0.89 | [0.869, 0.922] | 0.24 | -0.16 | 0.81 | 4.22 | 0.77 | 360 |

Judgments are clustered and report viewing follows the unaided rating; not an order-controlled causal effect. ΔConf is confidence change, not calibration.

### Semantic formalization fidelity by validator (from the audit)

| validator | n_audited | fidelity_accuracy |
|---|---|---|
| alp | 72 | 0.76 |
| problog | 65 | 0.75 |
| prolog | 55 | 0.87 |
| z3 | 58 | 0.84 |

This is measured fidelity to the source text, distinct from checker acceptance (Table 22).

### Annotation agreement (before adjudication)

| study | field | n_items | n_raters | raw_agreement | fleiss_kappa |
|---|---|---|---|---|---|
| original (6 annotators) | regime | 400 | 6 | 0.87 | 0.85 |
| original (6 annotators) | bridge_required | 400 | 6 | 0.82 | 0.45 |
| independent audit (2 experts) | regime | 250 | 2 | 0.91 | 0.89 |
| independent audit (2 experts) | faithful | 250 | 2 | 0.84 | 0.53 |

### Table 22. Native-checker acceptance and refinement retries (cap k = 5)

| validator | n_obligations | first_acceptance | final_acceptance | mean_retries_all | mean_retries_given_first_failure | n_failed_first | cap |
|---|---|---|---|---|---|---|---|
| alp | 600 | 0.84 | 0.95 | 0.53 | 3.29 | 97 | 5 |
| lean | 600 | 0.85 | 0.94 | 0.51 | 3.45 | 89 | 5 |
| problog | 600 | 0.80 | 0.98 | 0.48 | 2.47 | 118 | 5 |
| z3 | 600 | 0.80 | 0.98 | 0.45 | 2.24 | 121 | 5 |
| POOLED (instance-weighted) | 2400 | 0.82 | 0.96 | 0.49 | 2.80 | 425 | 5 |

Acceptance by the checker does not establish fidelity to the source statement.

### Table 18. Mean stage latency (s) and sequential sum

| system | decomp | native | bridges | counter_abd | aggregation | sequential_sum |
|---|---|---|---|---|---|---|
| Single-logic PCR | 0.21 | 0.31 | – | 0.25 | 0.19 | 0.96 |
| ValidLLP-style | 0.21 | 0.33 | – | 0.25 | 0.19 | 0.98 |
| MetaPCR w/o bridges | 0.21 | 0.34 | – | 0.25 | 0.19 | 0.99 |
| Full MetaPCR-LLM | 0.21 | 0.38 | 0.32 | 0.25 | 0.19 | 1.34 |

### Table 19. End-to-end wall-clock latency (s)

| system | mean | median | p95 | warning |
|---|---|---|---|---|
| Single-logic PCR | 0.74 | 0.74 | 0.85 | – |
| ValidLLP-style | 0.80 | 0.80 | 0.91 | – |
| MetaPCR w/o bridges | 0.82 | 0.82 | 0.93 | – |
| Full MetaPCR-LLM | 0.90 | 0.90 | 1.03 | – |

### Table 21. Manually analyzed failures

| category | count | pct |
|---|---|---|
| Semantic decomposition | 28 | 22.58 |
| Logic routing | 27 | 21.77 |
| Formalization | 21 | 16.94 |
| Cross-logic bridge | 26 | 20.97 |
| Native-validator limitation | 22 | 17.74 |
| Routing + bridge | 53 | 42.74 |
| Decomposition + formalization | 49 | 39.52 |
| Total | 124 | 100.00 |

### Table 16. Mixed-case screening ledger

| origin | examined | mixed | not_mixed | ambiguous | excluded | mixed_rate |
|---|---|---|---|---|---|---|
| released | 1500 | 420 | 950 | 85 | 45 | 0.28 |
| stress | 400 | 400 | 0 | 0 | 0 | 1.00 |
| transformed | 600 | 510 | 40 | 30 | 20 | 0.85 |

### Section 5.14. Empirical growth exponents (log-log slopes)

| strategy | varied | held fixed | quantity | log-log slope |
|---|---|---|---|---|
| independent | L | K=3, q=1, n=10 | candidates_per_graph | 1.08 |
| independent | L | K=3, q=1, n=10 | cross_logic_transitions_per_graph | 1.08 |
| unpruned | L | K=10, q=1, n=10 | candidates_per_graph | 1.06 |
| unpruned | L | K=10, q=1, n=10 | cross_logic_transitions_per_graph | 1.06 |
| beam | L | K=10, q=1, n=10 | candidates_per_graph | 0.39 |
| beam | L | K=10, q=1, n=10 | cross_logic_transitions_per_graph | 0.39 |
| independent | K | L=80, q=1, n=10 | candidates_per_graph | 1.00 |
| unpruned | K | L=80, q=1, n=10 | candidates_per_reached_transition | 2.22 |
| beam | K | L=80, q=1, n=10 | candidates_per_reached_transition | 0.16 |
| independent | q | K=1, L=80, n=10 | cross_logic_transitions_per_graph | 1.01 |

All Table 11 upper bounds are asserted on every configuration; full grid in scalability_grid.csv. Expected: ~1 in L; ~1 in K for independent chains; ~2 in K unpruned; ~0 in K with beam width w = 3; ~1 in q. Recomposition stops when invalid bridges empty the frontier, so per-graph totals for recomposition can grow sublinearly in L; per-reached-transition counts isolate the K dependence.

### Consistency of reported Table 20 p-values with their CIs

| name | diff | ci_low | ci_high | reported_p | implied_raw_p | implied_holm_p | log10_gap | consistent | family_rank_monotone |
|---|---|---|---|---|---|---|---|---|---|
| ProcessBench | 0.04 | 0.02 | 0.06 | 1.06e-07 | 3.66e-04 | 3.66e-04 | 3.54 | False | False |
| RAGTruth | 0.04 | 0.02 | 0.06 | 1.00e-16 | 3.69e-05 | 7.37e-05 | 11.87 | False | False |
| ProofWriter | 0.13 | 0.09 | 0.17 | 2.82e-07 | 1.47e-12 | 5.86e-12 | 4.68 | False | False |
| Pooled mixed | 0.11 | 0.08 | 0.14 | 4.12e-09 | 1.61e-11 | 4.84e-11 | 1.93 | False | False |

Implied p uses a normal approximation from the CI width; '<1e-16' is entered as 1e-16. A gap above 1 order of magnitude, or a non-monotone Holm ordering, needs explanation.

### Consistency of reported original-study p-values with their CIs

| name | diff | ci_low | ci_high | reported_p | implied_raw_p | implied_holm_p | log10_gap | consistent | family_rank_monotone |
|---|---|---|---|---|---|---|---|---|---|
| Primary accuracy | 0.03 | 0.02 | 0.04 | 1.06e-07 | 5.20e-08 | 1.04e-07 | 0.008 | True | True |
| Primary pooled F1 | 0.06 | 0.05 | 0.07 | 1.00e-16 | 4.97e-23 | 1.99e-22 | 5.70 | True | True |
| Mixed F1 vs ValidLLP | 0.06 | 0.04 | 0.08 | 2.82e-07 | 3.17e-07 | 3.17e-07 | 0.05 | True | True |
| Mixed F1 vs no-bridge | 0.10 | 0.08 | 0.12 | 3.33e-16 | 1.16e-16 | 3.47e-16 | 0.02 | True | True |

Implied p uses a normal approximation from the CI width; '<1e-16' is entered as 1e-16. A gap above 1 order of magnitude, or a non-monotone Holm ordering, needs explanation.
