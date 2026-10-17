# Reasoning Decomposition Configuration Archive
**Identifier:** `prompt-archive-v2.4-stable/decomp`

## Model & Snapshot
- **Model:** GPT-4o (`gpt-4o-2024-05-13`)
- **Temperature:** 0.0

## System Template
```text
You are a formal logical compiler. Your task is to segment a natural language reasoning chain into atomic, single-step inference steps. Each step must isolate its specific premises, the claimed relation (deduction, abduction, constraint, probability, temporal), and the immediate conclusion.
```
## User Prompt Template
```text
Decompose the following rationale into a structured JSON array matching this schema:
{
  "step_id": "sX",
  "premises": ["string"],
  "conclusion": "string",
  "claimed_relation": "string",
  "source_span": "string"
}

Rationale to decompose:
{rationale_text}
```
