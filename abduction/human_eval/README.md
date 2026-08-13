# Human Evaluation Package for Chapter 9

This package implements the two-stage human-evaluation protocol described in
the manuscript.

## Files

- `human_evaluation_form.py` — Streamlit form used by evaluators.
- `process_human_evaluation.py` — analysis script for the collected CSV log.
- `sample_evaluation_cases.csv` — 18 synthetic demo stimuli across the three
  manuscript datasets and six evaluated systems.
- `sample_evaluation_log.csv` — 216 synthetic demo ratings from 36 participant
  codes (12 clinicians, 12 AI researchers, 12 general users).
- `sample_processed_results/` — example outputs from processing the synthetic log.

**Important:** every sample row is marked `is_synthetic_demo=True`. These data
exist only to demonstrate the collection and processing workflow and should not
be reported as empirical results.

## Evaluator form

The form uses two phases.

### Phase 1: before the verifier explanation
The evaluator sees the context/evidence, question, and LLM answer, then records:
1. hallucination judgment: Hallucinated / Not hallucinated;
2. confidence in that judgment: 0–100.

The gold label and verification-system identity are hidden.

### Phase 2: after the verifier explanation
The evaluator sees the verification explanation and records:
1. clarity: 1–5;
2. coherence: 1–5;
3. trust: 1–5;
4. revised hallucination judgment;
5. revised confidence: 0–100;
6. optional useful-evidence tags and comments.

The 0–100 confidence values are normalized to [0,1] during analysis so the
before/after values can be reported in the same form as the manuscript's trust
calibration table.

## Recommended field definitions

- **Clarity:** how easy the verification explanation is to understand.
- **Coherence:** how logically connected and internally consistent the
  explanation is.
- **Trust:** how trustworthy the verifier's explanation appears to the
  evaluator.
- **Confidence before/after:** confidence in the evaluator's own hallucination
  judgment, not a direct rating of system quality.

## Running the form

```bash
pip install streamlit pandas
cd human_evaluation_package
streamlit run human_evaluation_form.py
```

The default output log is `human_evaluation_log.csv`. To write elsewhere:

```bash
HUMAN_EVAL_LOG=/path/to/log.csv streamlit run human_evaluation_form.py
```

Use participant codes rather than names or email addresses.

## Processing collected evaluations

```bash
python process_human_evaluation.py   --input human_evaluation_log.csv   --out-dir human_eval_results
```

The script produces system-, dataset-, and evaluator-group summaries, human
hallucination-judgment precision/recall/F1 before and after the explanation,
quality-control checks, and an overall JSON summary.

`chapter9_confidence_delta = confidence_after - confidence_before` reproduces
the simple pre/post confidence shift described in the manuscript. The script
also reports `correctness_aware_gain` as an additional diagnostic so increased
confidence in an incorrect final judgment is not interpreted as improved
calibration.
