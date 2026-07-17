# JKRHM / Discourse-CoT Hallucination Reproducibility Package

This package contains the YAML configuration and Python scripts referenced in the manuscript subsection on model settings, random seeds, and evaluation scripts.

## Files

- `config.yaml`: central configuration for paths, dataset sources, seeds, features, model settings, and thresholds.
- `scripts/build_dataset.py`: loads the GitHub/local dataset or creates a deterministic synthetic fallback dataset.
- `scripts/parse_discourse.py`: constructs a reproducible heuristic discourse/reasoning tree for each CoT trace.
- `scripts/extract_features.py`: extracts lexical/style, topology, discourse-relation, nucleus/satellite, domain-cue, and JKRHM proxy features.
- `scripts/run_evaluation.py`: trains and evaluates the classifier over repeated random splits.
- `scripts/run_ablations.py`: runs leakage checks and ablations for feature groups.
- `scripts/summarize_results.py`: reports mean, standard deviation, 95% CI, and paired tests.
- `prompts/discourse_parse_prompt.txt`: prompt template for replacing the heuristic parser with an LLM/RST parser.

## Quick start

```bash
pip install -r requirements.txt
python scripts/build_dataset.py --config config.yaml --fallback
python scripts/parse_discourse.py --config config.yaml
python scripts/extract_features.py --config config.yaml
python scripts/run_evaluation.py --config config.yaml
python scripts/run_ablations.py --config config.yaml
python scripts/summarize_results.py --config config.yaml
```

Remove `--fallback` to try the local files listed in `config.yaml` and then GitHub raw URLs. If GitHub raw files are unavailable, the fallback generator creates a deterministic 1,000-example dataset with the same canonical schema.

## Canonical dataset schema

`id, patient_complaint, diagnosis, reasoning_log, discourse_tree, hallucination_label`

`hallucination_label` is normalized to `0` for grounded and `1` for hallucinated.
