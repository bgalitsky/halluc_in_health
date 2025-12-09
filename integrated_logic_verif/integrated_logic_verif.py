"""
Pipeline with Prolog–Python bridge (pyswip):

1. get_llm_answer(query)
   -> baseline GPT-5 answer

2. get_logic_spec(query, answer)
   -> GPT-5 chooses logical formalism (LP / PLP / ALP / AF / ASP / etc.)
      and emits SWI-Prolog code + the name of a validation predicate

3. run_swi_prolog(spec)
   -> consults the Prolog code via pyswip and calls validation predicate

4. validate_with_logic(query)
   -> orchestrates everything and compares verdict vs. answer
"""

import os
import json
import tempfile
from typing import Dict, Any

from openai import OpenAI
from pyswip import Prolog

# Make sure OPENAI_API_KEY is set in the environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_llm_answer(query: str) -> str:
    """
    Ask GPT-5 for a concise answer to the user query.
    This is the answer we later try to logically validate.
    """
    response = client.responses.create(
        model="gpt-5.1",  # or your preferred GPT-5 snapshot
        input=(
            "You are a domain expert (medicine / law / finance / general knowledge).\n"
            "User question:\n"
            f"{query}\n\n"
            "Give a concise, factual answer suitable for downstream logical validation.\n"
            "Do not include any JSON, metadata, or commentary."
        ),
        max_output_tokens=512,
        reasoning={"effort": "medium"},
    )

    answer = response.output[0].content[0].text.strip()
    return answer


def get_logic_spec(query: str, answer: str) -> Dict[str, Any]:
    """
    Ask GPT-5 to:
      - SELECT one logical formalism from your list (LP, PLP, ALP, AF, DeLP, ASP, CLP, DL, ...)
      - BUILD a minimal ontology and rules in SWI-Prolog syntax
      - DEFINE a validation predicate that returns a status atom

    Contract (very important):

    - GPT must return JSON with fields:
        {
          "formalism": "<one of LP, LP_IC, PLP, ALP, AF, ABA, DeLP, ASP, CLP, DL>",
          "prolog_code": "<full SWI-Prolog program as a single string>",
          "validation_goal": "<predicate name of arity 1, e.g., 'validation_result'>"
        }

    - The Prolog program MUST define a predicate:
        validation_goal(Status)
      where Status is one of: supported, contradicted, unknown.
    """

    system_instructions = """
You are a logic engineer building SWI-Prolog validators for LLM answers.

You have the following menu of formalisms (pick ONE per request):

  - "LP"   : Horn-clause logic programming
  - "LP_IC": LP + explicit negation + integrity constraints
  - "PLP"  : probabilistic logic programming (approximate with deterministic rules if needed)
  - "ALP"  : abductive logic programming
  - "AF"   : abstract argumentation framework (Dung-style), encoded in Prolog
  - "ABA"  : assumption-based argumentation
  - "DeLP" : defeasible logic programming
  - "ASP"  : answer-set / stable-model style (implementable subset in Prolog)
  - "CLP"  : constraint logic programming (numeric / temporal)
  - "DL"   : description-logic-style ontology (encoded directly as Prolog predicates)

You must:

1. Choose ONE formalism name from the above list which is most appropriate
   for validating the given LLM answer.

2. Produce SWI-Prolog code in a string `prolog_code` that:
   - defines the domain ontology as facts and rules
   - defines a predicate claim_supported/0 that succeeds iff the LLM answer
     is supported/consistent/accepted under your chosen formalism
   - optionally defines claim_contradicted/0 if you want to detect explicit contradictions

3. Define a predicate of arity 1, e.g. validation_result/1, such that:
   - validation_result(Status) succeeds once
   - Status is one of the atoms: supported, contradicted, unknown
   - The logic for Status is:
       * supported      if claim_supported holds and no stronger contradiction exists
       * contradicted   if claim_contradicted holds
       * unknown        otherwise

4. Make the program self-contained: it must compile and run in SWI-Prolog
   without external files.

Return ONLY a JSON object with fields:

{
  "formalism": "<one of LP, LP_IC, PLP, ALP, AF, ABA, DeLP, ASP, CLP, DL>",
  "prolog_code": "<full SWI-Prolog program as a single string>",
  "validation_goal": "<predicate name of arity 1, e.g., 'validation_result'>"
}
"""

    user_prompt = f"""
User query:
{query}

LLM answer to validate:
{answer}

Task:
- Construct an appropriate symbolic representation of the domain.
- Encode it in SWI-Prolog using ONE of the above formalisms.
- Implement claim_supported/0 (and optionally claim_contradicted/0).
- Implement a predicate of arity 1 (e.g., validation_result/1) that unifies Status
  with one of the atoms: supported, contradicted, unknown.

Remember to return ONLY a single JSON object, no extra text.
"""

    response = client.responses.create(
        model="gpt-5.1",
        instructions=system_instructions,
        input=user_prompt,
        max_output_tokens=2048,
        reasoning={"effort": "high"},
    )

    raw = response.output[0].content[0].text.strip()

    # In practice you may want to add safety guards & retries here.
    spec = json.loads(raw)
    return spec


def run_swi_prolog(spec: Dict[str, Any]) -> str:
    """
    Run the generated Prolog code using SWI-Prolog via pyswip and capture the Status.

    We expect:
      - Prolog code is self-contained.
      - It defines a predicate `validation_goal(Status)` where Status is
        one of: supported, contradicted, unknown (atoms).
    """

    prolog_code = spec["prolog_code"]
    validation_goal = spec["validation_goal"]  # e.g., 'validation_result'

    # Write Prolog code to a temporary file and consult it
    with tempfile.NamedTemporaryFile(suffix=".pl", delete=False, mode="w", encoding="utf-8") as f:
        f.write(prolog_code)
        prolog_file = f.name

    prolog = Prolog()
    try:
        prolog.consult(prolog_file)
    except Exception as e:
        # Compilation/consult error
        return "prolog_error"

    # Run the query: validation_goal(Status)
    try:
        query = prolog.query(f"{validation_goal}(Status)")
        result = next(iter(query), None)
        query.close()
    except Exception:
        return "prolog_error"

    if result is None:
        return "unknown"

    status = str(result.get("Status", "unknown")).lower()
    if status not in {"supported", "contradicted", "unknown"}:
        return "unknown"

    return status


def validate_with_logic(query: str) -> Dict[str, Any]:
    """
    End-to-end pipeline:
      1. Get LLM answer
      2. Ask LLM to build Prolog validator (ontology + rules + validation predicate)
      3. Run SWI-Prolog via pyswip
      4. Compare verdict vs. answer
    """

    # 1) LLM answer
    answer = get_llm_answer(query)

    # 2) Logical spec (formalism + code + validation predicate name)
    logic_spec = get_logic_spec(query, answer)

    # 3) Execute SWI-Prolog validator via bridge
    verdict = run_swi_prolog(logic_spec)

    # 4) Simple interpretation – you can enrich this with more nuanced
    #    explanations, mapping to hallucination flags, etc.
    supported = verdict == "supported"
    contradicted = verdict == "contradicted"

    return {
        "query": query,
        "llm_answer": answer,
        "logic_formalism": logic_spec.get("formalism"),
        "prolog_verdict": verdict,
        "is_supported": supported,
        "is_contradicted": contradicted,
        "logic_spec": logic_spec,  # keep full spec for debugging / logging
    }


if __name__ == "__main__":
    # Example usage
    example_query = "Can walking in cold sea water directly cause gout in a healthy adult?"
    result = validate_with_logic(example_query)

    print("=== Logical Validation Result ===")
    print("Formalism:     ", result["logic_formalism"])
    print("LLM answer:    ", result["llm_answer"])
    print("Prolog verdict:", result["prolog_verdict"])
    print("Supported?     ", result["is_supported"])
    print("Contradicted?  ", result["is_contradicted"])
