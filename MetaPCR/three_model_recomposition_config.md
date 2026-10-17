# Three-Model Recomposition Config
**Identifier:** `prompt-archive-v2.4-stable/recompose`

## Co-Contributing Models & Snapshots
1. **Model A:** GPT-4o (`gpt-4o-2024-05-13`) | Temperature: 0.7
2. **Model B:** Claude 3.5 Sonnet (`claude-3-5-sonnet-20240620`) | Temperature: 0.5
3. **Model C:** Llama 3.1 405B Instruct (`llama-3.1-405b-instruct`) | Temperature: 0.6

## Recomposition Framework
Extracts certified steps from all three model logs, maps premise dependencies, and identifies candidate cross-logic logic graphs using bounded-beam pruning ($w=3$).
