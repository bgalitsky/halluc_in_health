# Answer Generation Configuration Archive
**Identifier:** `prompt-archive-v2.4-stable/answer_gen`

## Model & Snapshot
- **Model:** GPT-4o (`gpt-4o-2024-05-13`)
- **Temperature:** 0.7

## System Template
```text
You are an expert domain reasoner. Provide a detailed, step-by-step rationale before concluding with the final answer. Ensure that intermediate claims, mathematical steps, and deductive leaps are explicitly stated.
```
## User Prompt Template
```text
[Context / Available Evidence]
{evidence_context}

[Query]
{query_text}

Provide your complete multi-step reasoning path below, bounded clearly by individual inferential links.
```
