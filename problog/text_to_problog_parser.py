"""
Full pipeline:
1. Get discourse tree from your FastAPI parser.
2. Send the JSON + probability policy to GPT-4.
3. GPT-4 returns a ProbLog program with p::facts and p::rules.
4. Evaluate the program with ProbLog.

Prereqs:
    pip install requests openai problog
Env:
    export OPENAI_API_KEY=...
"""

import json
import requests
from openai import OpenAI
from problog.program import PrologString
from problog.sdd_formula import SDD
from problog import get_evaluatable
import os
from functools import lru_cache
from joblib import Memory
from configparser import RawConfigParser
from problog.core import ProbLog

# Cache directory persists between runs
#memory = Memory("./cache", verbose=1)
memory = Memory("./llm_prolog_cache", verbose=1, compress=1)

# Get directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build full path to config.ini relative to this script
config_path = os.path.join(script_dir, 'config.ini')


# Load configuration
config = RawConfigParser()
config.read(config_path) #'config.ini')

# API keys

os.environ["OPENAI_API_KEY"] = config.get('OpenAI', 'api_key')


MODEL_NAME = "gpt-5"  # or "gpt-3.5-turbo"

# -------------------------------
# Config: probability policy
# -------------------------------
# Base probabilities by discourse role
ROLE_BASE = {
    "nucleus": 0.78,    # ядро
    "satellite": 0.58   # сателлит
}

# Relation adjustments (positive means stronger, negative weaker)
RELATION_ADJ = {
    "cause": +0.08,
    "result": +0.06,
    "evidence": +0.07,
    "explanation": +0.06,
    "purpose": +0.05,
    "sequence": 0.00,
    "background": -0.01,
    "elaboration": -0.04,
    "circumstance": -0.03,
    "concession": -0.05,
    "enablement": -0.02
}

# -------------------------------
# 1. Get discourse tree from FastAPI
# -------------------------------
@lru_cache(maxsize=128)
@memory.cache
def get_discourse_tree(text: str,
                       url: str = "http://54.82.56.2:8000/analyze") -> dict:
    """Calls your FastAPI discourse parser to get JSON."""
    resp = requests.post(url, json={"text": text})
    resp.raise_for_status()
    return resp.json()

# -------------------------------
# 2. Convert JSON to ProbLog via GPT-4
# -------------------------------
client = OpenAI()

#@lru_cache(maxsize=128)
#@memory.cache
def discourse_to_problog(discourse_json: dict) -> str:
    """
    Ask GPT-4 to produce a ProbLog program using the discourse JSON
    and the probability policy.
    """
    system_prompt = (
        "You are an expert in probabilistic logic programming. "
        "You receive JSON from a discourse parser: facts, rules, queries. "
        "Each fact has an atom, a discourse role (nucleus or satellite) and a rhetorical relation. "
        "Each rule has head, body, role, and relation. "
        "Use the following probability policy:\n\n"
        f"Base probabilities by role: {json.dumps(ROLE_BASE)}\n"
        f"Relation adjustments: {json.dumps(RELATION_ADJ)}\n\n"
        "Assign p = clamp(BASE(role) + ADJ(relation), 0.05, 0.95) to each fact and rule. "
        "Return a valid ProbLog program ONLY (no explanations). "
        "Facts: 'p::atom.' Rules: 'p::head :- body.' Add queries at the end."
    )

    user_prompt = (
        "Here is the discourse tree JSON:\n\n"
        f"{json.dumps(discourse_json, ensure_ascii=False, indent=2)}\n\n"
        "Produce the ProbLog program now."
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return resp.choices[0].message.content.strip()

# -------------------------------
# 3. Evaluate ProbLog
# -------------------------------
def evaluate_problog(problog_program: str):
    # ProbLog.convert(p, SDD).evaluate())

    model = PrologString(problog_program)
    result = ProbLog.convert(model, SDD).evaluate()

    print(result)
  #  result = get_evaluatable(SDD).create_from(model).evaluate()
    return result #dict(sorted(result.items(), key=lambda kv: kv[0]))

#ProbLog.convert(p, SDD).evaluate())

# -------------------------------
# 4. Run the pipeline
# -------------------------------
if __name__ == "__main__":
    text = ("For the past few days in my knee and ankle pain was throbbing. "
            "When I woke up at night due to severe pain, I took a painkiller. "
            "When I carefully looked at my red joint, it seemed swollen. "
            "Once I discovered that I had a fever, I started thinking how to cure it.")

    # Step 1: get discourse tree JSON from FastAPI
    discourse_json = get_discourse_tree(text)

    # Step 2: ask GPT-4 to convert JSON to ProbLog program
    problog_program = discourse_to_problog(discourse_json)

    print("\n=== ProbLog program from GPT-4 ===\n")
    print(problog_program)

    # Step 3: evaluate the ProbLog program
    try:
        res = evaluate_problog(problog_program)
        print("\n=== Evaluation ===\n")
        for atom, prob in res.items():
            print(f"{str(atom):40s}  {prob:.4f}")
    except Exception as e:
        print(f"Evaluation error: {e}")
