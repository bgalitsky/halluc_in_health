# Hall2Invent abductive approval pipeline

This program applies the paper's governed hallucination-to-invention workflow
to `halluc2invention_1000_with_answers.json`.

## What it does

For each record it:

1. relabels unsupported claims as low-prior abductive hypotheses;
2. extracts the design goal, observations, assumptions, and claims;
3. checks physical, logical, safety, regulatory, resource, operational,
   ethical, and evidential integrity constraints;
4. performs counter-abduction through mechanism substitution, feature
   preservation, goal decomposition, assumption revision, and boundary
   construction;
5. ranks hypotheses using a weighted abductive objective;
6. optionally runs an independent second-pass critic;
7. returns `approve` or `reject`, a validation plan, constraint failures,
   provenance, and a human-review flag;
8. compares the decision with the dataset's `acceptability` field without
   revealing that field to the model.

`approve` means **retain as an invention candidate for further review**. It is
not a patentability opinion or permission to deploy the design.

## Important evaluation choice

The generated dataset's `invention_description` may contain explicit verdict
language such as “not technically acceptable.” To avoid label leakage, the
script defaults to:

```bash
--candidate-source hallucinated_answer
```

Use `--candidate-source invention_description` only for operational review,
not for a fair benchmark against the existing labels.

The model also does **not** receive `acceptability` or `trigger_type` by
default. `trigger_type` can be enabled only for an explicit ablation with
`--include-trigger-types`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements_hall2invent.txt
export OPENAI_API_KEY="..."
```

## Test five records first

```bash
python hall2invent_abductive_approval.py \
  --input halluc2invention_1000_with_answers.json \
  --max-cases 5 \
  --mode full
```

The default model is `gpt-5-mini`. For a stronger but more expensive run:

```bash
python hall2invent_abductive_approval.py \
  --input halluc2invention_1000_with_answers.json \
  --model gpt-5.2 \
  --critic-model gpt-5.2 \
  --mode full \
  --workers 2
```

## Run the full dataset

```bash
python hall2invent_abductive_approval.py \
  --input halluc2invention_1000_with_answers.json \
  --output hall2invent_predictions.jsonl \
  --summary hall2invent_summary.json \
  --csv hall2invent_predictions.csv \
  --mode full \
  --workers 4
```

The JSONL file is append-only and acts as a resume cache. Re-running the same
command skips successfully completed IDs.

## Optional preliminary web grounding

```bash
python hall2invent_abductive_approval.py \
  --input halluc2invention_1000_with_answers.json \
  --web-search \
  --save-evidence \
  --max-cases 10
```

Web retrieval helps identify known limits and prior-art categories, but it is
not a complete patent search, regulatory review, safety case, simulation, or
experiment.

## Fast, lower-cost mode

```bash
python hall2invent_abductive_approval.py \
  --input halluc2invention_1000_with_answers.json \
  --mode fast \
  --workers 4
```

`fast` skips the independent critic. `full` is closer to the paper's
human-supervised counter-abductive workflow.

## Main outputs

- `hall2invent_predictions.jsonl`: full per-case analyses and decisions
- `hall2invent_predictions.csv`: compact decisions
- `hall2invent_summary.json`: counts, confusion matrix, accuracy, precision,
  recall, F1, specificity, elapsed time, and token usage

## Reproducibility notes

For a paper-quality experiment, record:

- exact model snapshot rather than only an alias;
- prompts and script commit;
- API access date;
- thresholds and objective weights;
- number of retries and workers;
- whether web search was enabled;
- randomization and test split;
- expert-review procedure;
- confidence intervals and statistical tests.

The OpenAI model provides structured judgments, but the final decision gates
are deterministic and visible in the script. This prevents the model from
silently changing the definition of approval.
