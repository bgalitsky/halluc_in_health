# D-ALP Abductive RAG — Section 6 code

`d_alp_rag.py` implements the Section-6 inference pipeline.

`learn_weights.py` learns:
- rhetorical-sufficiency weights for relevance/coherence/coverage,
- threshold tau,
- contradiction and defeat penalties,
- final candidate-ranking weights for explanatory adequacy/grounding/discourse.

## Train weights

```bash
python learn_weights.py   --rhetoric-csv rhetoric_training_example.csv   --candidate-csv candidate_training_example.csv   --out learned_weights.json
```

## Training labels

Rhetorical data:
`relevance, coherence, coverage, sufficient`

Candidate data:
`query_id, candidate_id, goal_fit, contradiction, defeat, grounding, discourse_weight, preferred`

For each `query_id`, include at least one preferred and one rejected candidate.

## Model adapters

The inference code is model-agnostic. Supply:
- `LLM(prompt) -> str`
- `NLIScorer(premise, hypothesis) -> probabilities`
- `Retriever.search(query, k) -> [(passage, score)]`

Replace the lightweight lexical fallback functions with the actual D-ALP discourse,
coverage, symbolic consistency, and counter-abductive modules used in experiments.
