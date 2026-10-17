# Autoformalization and Iterative Repair Config
**Identifier:** `prompt-archive-v2.4-stable/autoformalize_repair`

## Model & Snapshot
- **Model:** GPT-4o (`gpt-4o-2024-05-13`)
- **Temperature:** 0.1

## Target Logic Routing & Templates
- **Lean 4:** Encodes functional math proofs.
- **Z3 (SMT):** Encodes linear arithmetic and boundary constraints.
- **SWI-Prolog (ALP):** Encodes background rules, integrity constraints, and abducible declarations.
- **ProbLog:** Encodes uncertain relational rules with explicit weights.

## Iterative Repair Protocol (Max 5 Retries)
```text
The solver returned the following error message while parsing your formal translation:
{solver_error_message}

Review the syntax constraints of {target_logic}. Correct the error while strictly preserving the original natural-language semantics. Output ONLY the raw executable code blocks.
```
