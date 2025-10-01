"""
Pipeline:
- Take two texts: one narrative (facts) and one definitional (clauses).
- Call FastAPI discourse parser for each.
- GPT-4 converts each discourse JSON to ProbLog program with probabilities.
- Merge the two ProbLog programs.
- Extract the head of the last rule from the clauses program to form a query.
- Evaluate the merged ProbLog program.
"""

import json
import re
import requests
from openai import OpenAI
from problog.program import PrologString
from problog.sdd_formula import SDD
from problog import get_evaluatable
from problog.core import ProbLog
import os
from joblib import Memory
from configparser import RawConfigParser
from problog.core import ProbLog
from multiprocessing import Process, Queue

def eval_worker(program: PrologString, queue: Queue):
    try:
        result1 = ProbLog.convert(program, SDD).evaluate()
        queue.put(result1)
    except Exception as e:
        queue.put(e)

def evaluate_with_timeout(program: PrologString, timeout: int = 60):
    q = Queue()
    p = Process(target=eval_worker, args=(program, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return {}  # empty result on timeout
    res = q.get()
    if isinstance(res, Exception):
        raise res
    return res


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


MODEL_NAME = "gpt-5"
# -------------------------------
# Config: probability policy
# -------------------------------
ROLE_BASE = {"nucleus": 0.78, "satellite": 0.58}
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

import re

def analyze_ontology(ontology: str):
    """
    Parse a ProbLog ontology (probabilistic logic program) and return:
    1) Comma-separated set of body predicate names (excluding the last clause)
    2) Head of the last clause in the form predicate(D)
    """
    # split into meaningful clauses, skip comments and queries
    clauses = [
        line.strip().rstrip('.')
        for line in ontology.splitlines()
        if line.strip() and not line.strip().startswith("%") and not line.strip().startswith("query")
    ]

    body_preds = []

    for clause in clauses[:-1]:
        # remove any leading probability like "0.8::"
        clause_no_prob = re.sub(r'^\s*\d*\.?\d+\s*::\s*', '', clause)
        if ":-" in clause_no_prob:
            body = clause_no_prob.split(":-", 1)[1]
            # collect all functor names in body atoms
            preds = re.findall(r'([a-zA-Z_]\w*)\s*\(', body)
            body_preds.extend(preds)

    # process last clause head
    last_clause = clauses[-1]
    last_clause_no_prob = re.sub(r'^\s*\d*\.?\d+\s*::\s*', '', last_clause)

    # grab head functor (everything before '(' or before ':-')
    head_pred = None
    m = re.match(r'\s*([a-zA-Z_]\w*)\s*\(', last_clause_no_prob)
    if m:
        head_pred = m.group(1)
    else:
        # maybe head without parentheses
        m2 = re.match(r'\s*([a-zA-Z_]\w*)', last_clause_no_prob)
        if m2:
            head_pred = m2.group(1)

    list_str = ', '.join(sorted(set(body_preds)))

    return list_str, f"{head_pred}(D)" if head_pred else None


# -------------------------------
# FastAPI discourse parser
# -------------------------------
def get_discourse_tree(text: str,
                       url: str = "http://54.82.56.2:8000/analyze") -> dict:
    """Calls your FastAPI discourse parser to get JSON."""
    resp = requests.post(url, json={"text": text})
    resp.raise_for_status()
    return resp.json()

# -------------------------------
# GPT-4 conversion
# -------------------------------
client = OpenAI()

def discourse_to_problog(discourse_json_ontology: dict, discourse_json_facts: dict) -> tuple[str,str]:
    """Ask GPT-4 to produce a ProbLog program given discourse JSON and input_type."""

    base_policy = (
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



    type_instructions_ontology = (
            "The JSON comes from a definitional or rule text (clauses). "
            "Produce probabilistic rules/clauses directly using p::head :- body. "
            "Add queries at the end if natural." +
            """
                    Example:
             % ontology text
            In most cases, only one or a few joints are affected. The big toe, knee, or ankle joints are most often affected.
            Sometimes many joints become swollen and painful.
            The pain starts suddenly, often during the night. Pain is often severe, described as throbbing, crushing, or excruciating.
            The joint appears warm and red. It is most often very tender and swollen (it hurts to put a sheet or blanket over it).
            There may be a fever.
            The attack may go away in a few days, but may return from time to time. Additional attacks often last longer.
            
             % Ontology rules for gout
            
            0.32::inflammation(joints(A)) :- joints(A), member(A, [one, few, both, multiple, toe, knee, ankle]).
            0.33::inflammation(pain(S)) :- pain(S), member(S, [painfull, severe, throbbing, crushing, excruciating]).
            0.73::inflammation(property(C)) :- property(C), member(C, [red, warm, tender, swollen, fever]).
            0.53::inflammation(last(L)) :- last(L), member(L, [few_days, return, additional(longer)]).
            
            % Disease definition
            disease(gout) :- inflammation(joints(A)),  inflammation(pain(S)), inflammation(property(C)), inflammation(last(L)).
            """

        )
    system_prompt_ontology = (
        "You are an expert in probabilistic logic programming. "
        "You receive JSON from a discourse parser with rules and queries. "
        f"{base_policy}{type_instructions_ontology} "
        "Return a valid ProbLog program ONLY (no explanations)."
    )

    user_prompt_ontology = (
        "Here is the discourse tree JSON:\n\n"
        f"{json.dumps(discourse_json_ontology, ensure_ascii=False, indent=2)}\n\n"
        "Produce the ProbLog program now."
    )

    print("going to gpt with ontology...")
    rsp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=1,
        messages=[{"role": "system", "content": system_prompt_ontology},
                  {"role": "user", "content": user_prompt_ontology}]
    )

    ontology_prog =  rsp.choices[0].message.content.strip()

    list_of_predicates, goal_predicate = analyze_ontology(ontology_prog)

    type_instructions_facts = (
            "The JSON comes from a narrative complaint (facts). "
            f"Produce probabilistic facts using predicates from the list {list_of_predicates}"
            "Use p::fact. For example, 'For the past few days in my knee and ankle pain was throbbing. "
                  "When I woke up at night due to severe pain, I took a painkiller. "
                  "When I carefully looked at my red joint, it seemed swollen. "
                  "Once I discovered that I had a fever, I started thinking how to cure it.'"+
    """ 
    and ontology: 
    0.74::inflammation(joints(A)) :- joints(A), member(A, [one, few, both, multiple, toe, knee, ankle]).
    0.74::inflammation(pain(S)) :- pain(S), member(S, [painful, severe, throbbing, crushing, excruciating]).
    0.74::inflammation(property(C)) :- property(C), member(C, [redness, warmth, tenderness, swelling, fever]).
    0.74::inflammation(duration(D)) :- duration(D), member(D, [few_days, recur, prolonged]).
    0.74::disease(gout) :- inflammation(joints(A)), inflammation(pain(S)), inflammation(property(C)), inflammation(duration(D)).
    % Disease definition
    disease(gout) :- inflammation(joints(A)),  inflammation(pain(S)), inflammation(property(C)), inflammation(last(L)).
    We have probabilistic logic program 
    0.71::joints(toe). 
    0.76::pain(severe). 
    0.34::property(fever). 
    0.21::last(return).
    """
    )

    system_prompt_facts = (
        "You are an expert in probabilistic logic programming. "
        "You receive JSON from a discourse parser with facts "
        f"{base_policy}{type_instructions_facts} "
        "Return a valid ProbLog program consisting from facts ONLY (no explanations)."
    )

    user_prompt_facts = (
        "Here is the discourse tree JSON:\n\n"
        f"{json.dumps(discourse_json_facts, ensure_ascii=False, indent=2)}\n\n"
        "Produce the ProbLog program now."
    )

    print("going to gpt with facts...")
    rsp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=1,
        messages=[{"role": "system", "content": system_prompt_facts},
                  {"role": "user", "content": user_prompt_facts}]
    )
    facts_prog = rsp.choices[0].message.content.strip()

    init_progr = """% add a deterministic definition:
        member(X,[X|_]).
        member(X,[_|T]) :- member(X,T)."
        \n"""

    return_prog = (ontology_prog + "\n" + facts_prog).replace("prolog```", "")
    return_prog = return_prog.replace("```", "")
    return return_prog, f"query({goal_predicate}"

# -------------------------------
# Evaluate ProbLog
# -------------------------------
def evaluate_problog(problog_program: str):

    # ProbLog.convert(p, SDD).evaluate())

    model = PrologString(problog_program)
    result = ProbLog.convert(model, SDD).evaluate()

    print(result)
  #  result = get_evaluatable(SDD).create_from(model).evaluate()
    return result #dict(sorted(result.items(), key=lambda kv: kv[0]))


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Text 1: facts narrative
    text_facts = ("For the past few days in my knee and ankle pain was throbbing. "
                  "When I woke up at night due to severe pain, I took a painkiller. "
                  "When I carefully looked at my red joint, it seemed swollen. "
                  "Once I discovered that I had a fever, I started thinking how to cure it.")
    # Text 2: clauses definition
    text_clauses = ("Gout as a disease characterized by inflammation that must simultaneously involve four dimensions: "
                    "(1) the joints affected, which may include one, a few, both, or multiple joints such as the toe, knee, or ankle; "
                    "(2) the nature of pain, which can be painful, severe, throbbing, crushing, or excruciating; "
                    "(3) observable inflammatory properties, such as redness, warmth, tenderness, swelling, or fever; and "
                    "(4) the temporal pattern, which may last a few days, recur, or become prolonged. "
                    "When inflammation is confirmed across these categories of joints, pain, properties, and duration, "
                    "the condition is classified as gout.")

    # Step 1: discourse tree JSON for each
    print("going to discourse parser...")
    disc_facts = get_discourse_tree(text_facts)
    disc_clauses = get_discourse_tree(text_clauses)

    # Step 2: GPT-4 to ProbLog

    merged_prog, extracted_query  = discourse_to_problog(disc_clauses,  disc_facts)

    print("\n=== Merged ProbLog program ===\n")
    print(merged_prog)
    print("\nExtracted query:", extracted_query)

    # Step 4: evaluate
    try:
        #res = evaluate_problog(merged_prog)
        res = evaluate_with_timeout(PrologString(merged_prog))
        print("\n=== Evaluation ===\n")
        for atom, prob in res.items():
            print(f"{str(atom):40s}  {prob:.4f}")
    except Exception as e:
        print(f"Evaluation error: {e}")
