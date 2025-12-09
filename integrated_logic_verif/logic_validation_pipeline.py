"""
End-to-end LLM + logic validation pipeline with real backends
and wired-in Prolog meta-interpreters:

- alp_engine.pl   (ALP)
- af_engine.pl    (AF)
- aba_engine.pl   (ABA)
- delp_engine.pl  (DeLP)
"""

import os
import json
import tempfile
from configparser import RawConfigParser
from dataclasses import dataclass
from typing import Dict, Any, List, Literal

from openai import OpenAI
from pyswip import Prolog  # SWI-Prolog bridge

# For ASP (clingo):
try:
    import clingo  # pip install clingo
    HAS_CLINGO = True
except ImportError:
    HAS_CLINGO = False


# ============================================
# 1. Formalism configuration table
# ============================================

BackendType = Literal["prolog", "asp"]


@dataclass
class FormalismConfig:
    name: str
    backend: BackendType
    python_deps: List[str]
    prolog_preamble: str = ""
    notes: str = ""


FORMALISM_CONFIGS: Dict[str, FormalismConfig] = {
    # Pure LP: nothing special
    "LP": FormalismConfig(
        name="Horn-Clause Logic Programming",
        backend="prolog",
        python_deps=["pyswip"],
        prolog_preamble="",
        notes="Plain SWI-Prolog; no extra packs.",
    ),
    # LP + integrity constraints / explicit negation
    "LP_IC": FormalismConfig(
        name="LP with Integrity Constraints",
        backend="prolog",
        python_deps=["pyswip"],
        prolog_preamble="",
        notes="Constraints implemented directly in SWI-Prolog.",
    ),
    # PLP via cplint / PITA
    "PLP": FormalismConfig(
        name="Probabilistic Logic Programming",
        backend="prolog",
        python_deps=["pyswip"],
        prolog_preamble=(
            "% Probabilistic logic programming (requires cplint/PITA pack).\n"
            ":- use_module(library(pita)).\n"
            ":- pita.\n"
        ),
        notes="Install cplint in SWI: ?- pack_install(cplint).",
    ),
    # Abductive logic programming via alp_engine.pl
    "ALP": FormalismConfig(
        name="Abductive Logic Programming",
        backend="prolog",
        python_deps=["pyswip"],
        prolog_preamble=(
            "% Abductive Logic Programming engine.\n"
            ":- use_module(alp_engine).\n"
        ),
        notes="alp_engine.pl must be in the same directory or on the Prolog library path.",
    ),
    # Abstract argumentation via af_engine.pl
    "AF": FormalismConfig(
        name="Abstract Argumentation Frameworks",
        backend="prolog",
        python_deps=["pyswip"],
        prolog_preamble=(
            "% Abstract argumentation (Dung AF) engine.\n"
            ":- use_module(af_engine).\n"
        ),
        notes="af_engine.pl must be available to SWI-Prolog.",
    ),
    # Assumption-based argumentation via aba_engine.pl
    "ABA": FormalismConfig(
        name="Assumption-Based Argumentation",
        backend="prolog",
        python_deps=["pyswip"],
        prolog_preamble=(
            "% Assumption-Based Argumentation engine.\n"
            ":- use_module(aba_engine).\n"
        ),
        notes="aba_engine.pl must be available to SWI-Prolog.",
    ),
    # Defeasible logic programming via delp_engine.pl
    "DeLP": FormalismConfig(
        name="Defeasible Logic Programming",
        backend="prolog",
        python_deps=["pyswip"],
        prolog_preamble=(
            "% Defeasible Logic Programming (simplified DeLP) engine.\n"
            ":- use_module(delp_engine).\n"
        ),
        notes="delp_engine.pl must be available to SWI-Prolog.",
    ),
    # ASP: clingo backend
    "ASP": FormalismConfig(
        name="Answer Set Programming (ASP)",
        backend="asp",
        python_deps=["clingo"],
        prolog_preamble="",
        notes="Uses clingo Python API for stable-model semantics.",
    ),
    # Constraint logic programming
    "CLP": FormalismConfig(
        name="Constraint Logic Programming",
        backend="prolog",
        python_deps=["pyswip"],
        prolog_preamble=(
            "% Constraint Logic Programming over finite domains & reals.\n"
            ":- use_module(library(clpfd)).\n"
            "% :- use_module(library(clpr)).\n"
        ),
        notes="Use clpfd/clpr for numeric/temporal constraints.",
    ),
    # Description logic / ontology
    "DL": FormalismConfig(
        name="Description Logic / Ontology Reasoning",
        backend="prolog",
        python_deps=["pyswip"],
        prolog_preamble=(
            "% Ontology / DL support; hook Thea or custom ontology modules here.\n"
            "% :- use_module(library(thea2_owl_parser)).\n"
        ),
        notes="You can integrate Thea or another OWL/DL reasoner.",
    ),
}


# ============================================
# 2. OpenAI client
# ============================================
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.ini')
config = RawConfigParser()
config.read(config_path) #'config.ini')
client = OpenAI(api_key=config.get('OpenAI', 'api_key'))


# ============================================
# 3. LLM answer (baseline)
# ============================================

def get_llm_answer(query: str) -> str:
    """
    Get a single concise LLM answer that will not break when response blocks differ in structure.
    """

    resp = client.responses.create(
        model="gpt-5.1",
        input=(
            "You are a domain expert. "
            "Answer the question factually and concisely.\n\n"
            f"Q: {query}\nA:"
        ),
        max_output_tokens=300,
        reasoning={"effort": "medium"},
    )


    answer = None

    try:
        for block in resp.output:
            # Some blocks contain 'content' with text segments, some don't
            if hasattr(block, "content") and block.content:
                for piece in block.content:
                    if hasattr(piece, "text") and piece.text and piece.text.strip():
                        answer = piece.text.strip()
                        break
            if answer:
                break
    except:
        pass

    # Safe fallback if no text returned at all
    if not answer:
        answer = "No answer."  # or raise error/log if desired

    return answer



# ============================================
# 4. Logical spec from GPT-5
# ============================================

def get_logic_spec(query: str, answer: str) -> Dict[str, Any]:
    """
    Ask GPT-5 to:
      - SELECT one logical formalism (LP, LP_IC, PLP, ALP, AF, ABA, DeLP, ASP, CLP, DL)
      - BUILD a minimal ontology and rules in the appropriate framework

    Contract:

    For PROLOG-based backends (LP, LP_IC, PLP, ALP, AF, ABA, DeLP, CLP, DL):

    - JSON:
        {
          "formalism": "<one of LP, LP_IC, PLP, ALP, AF, ABA, DeLP, CLP, DL>",
          "backend": "prolog",
          "program": "<SWI-Prolog program as a single string>",
          "validation_goal": "<predicate name of arity 1, e.g., 'validation_result'>"
        }

    - Prolog program MUST define predicate:
        validation_goal(Status)
      where Status ∈ {supported, contradicted, unknown}.

    For ASP (clingo):

    - JSON:
        {
          "formalism": "ASP",
          "backend": "asp",
          "program": "<ASP (clingo) program>",
          "asp_query": "<atom or scheme that determines Status>"
        }

      To keep things simple, we will assume you encode the final decision
      by deriving exactly one of the facts:
         status(supported).  / status(contradicted). / status(unknown).
    """

    system_instructions = """
You are a logic engineer building validators for LLM answers.

You have the following menu of formalisms:

  - "LP"   : Horn-clause logic programming
  - "LP_IC": LP + explicit negation + integrity constraints
  - "PLP"  : probabilistic logic programming (approximate with deterministic rules if needed)
  - "ALP"  : abductive logic programming (use module alp_engine)
  - "AF"   : abstract argumentation framework (Dung-style, use af_engine)
  - "ABA"  : assumption-based argumentation (use aba_engine)
  - "DeLP" : defeasible logic programming (use delp_engine)
  - "ASP"  : answer-set / stable-model style (clingo)
  - "CLP"  : constraint logic programming (numeric / temporal)
  - "DL"   : description-logic-style ontology (encoded in Prolog)

You must:

1. Choose ONE formalism name that is most appropriate for validating the given LLM answer.

2. If you choose any formalism EXCEPT "ASP":
   - Set "backend" to "prolog".
   - Produce SWI-Prolog code in field "program" that:
       * defines the domain ontology as facts and rules
       * defines a predicate claim_supported/0
       * optionally defines claim_contradicted/0
       * defines a predicate of arity 1, e.g., validation_result/1, such that:
           - validation_result(Status) succeeds once
           - Status is one of the atoms: supported, contradicted, unknown
           - Status is 'supported' if claim_supported holds and no stronger contradiction exists
           - Status is 'contradicted' if claim_contradicted holds
           - Status is 'unknown' otherwise

   - Set "validation_goal" to the name of that predicate (e.g., "validation_result").

   You MAY call into the following engines depending on your formalism:
     - ALP: use alp_engine: abduce/3, solve/3, check_ics/1
     - AF:  use af_engine: grounded_extension/1, in_grounded/1
     - ABA: use aba_engine: supported_in_ABA/1
     - DeLP: use delp_engine: warranted/1

3. If you choose "ASP":
   - Set "backend" to "asp".
   - Produce ASP (clingo) code in field "program".
   - Make sure your program derives exactly one fact:
         status(supported).  or
         status(contradicted). or
         status(unknown).
   - Set "asp_query" to the atom "status(Status)" such that we can inspect Status.

Return ONLY a JSON object with fields:

  - formalism
  - backend
  - program
  - validation_goal  (for prolog backends)
  - asp_query        (for ASP backend)
"""

    user_prompt = f"""
User query:
{query}

LLM answer to validate:
{answer}

Task:
- Construct an appropriate symbolic representation of the domain
  using ONE of the above formalisms.
- Follow the contract described in the system instructions.
- Return only the JSON object, no extra commentary.
"""

    response = client.responses.create(
        model="gpt-5.1",
        instructions=system_instructions,
        input=user_prompt,
        max_output_tokens=4096,
        reasoning={"effort": "high"},
    )

    raw = None
    try:
        for block in response.output:
            # Some blocks contain 'content' with text segments, some don't
            if hasattr(block, "content") and block.content:
                for piece in block.content:
                    if hasattr(piece, "text") and piece.text and piece.text.strip():
                        raw = piece.text.strip()
                        break
            if raw:
                break
    except:
        pass

    # Safe fallback if no text returned at all
    if not raw:
        return None
    spec = json.loads(raw)
    return spec


# ============================================
# 5. Prolog backend (pyswip)
# ============================================

def run_prolog_backend(spec: Dict[str, Any]) -> str:
    """
    Execute SWI-Prolog program using pyswip and read Status.

    - Injects per-formalism preamble (use_module/1, etc.).
    - Expects spec["program"] to define validation_goal(Status).
    """
    formalism = spec["formalism"]
    cfg = FORMALISM_CONFIGS.get(formalism)

    program_body = spec["program"]
    validation_goal = spec["validation_goal"]  # e.g., 'validation_result'

    # Inject preamble (libraries, engines, etc.)
    prolog_code = (cfg.prolog_preamble if cfg else "") + "\n\n" + program_body

    with tempfile.NamedTemporaryFile(suffix=".pl", delete=False, mode="w", encoding="utf-8") as f:
        f.write(prolog_code)
        prolog_file = f.name

    prolog = Prolog()
    try:
        prolog.consult(prolog_file)
    except Exception as e:
        print("Prolog consult error:", e)
        return "prolog_error"

    # Query: validation_goal(Status)
    try:
        q = prolog.query(f"{validation_goal}(Status)")
        result = next(iter(q), None)
        q.close()
    except Exception as e:
        print("Prolog query error:", e)
        return "prolog_error"

    if result is None:
        return "unknown"

    status = str(result.get("Status", "unknown")).lower()
    if status not in {"supported", "contradicted", "unknown"}:
        return "unknown"

    return status


# ============================================
# 6. ASP backend (clingo)
# ============================================

def run_asp_backend(spec: Dict[str, Any]) -> str:
    """
    Execute ASP (clingo) program and interpret derived status(Status).

    - Spec["program"]: ASP code
    - Spec["asp_query"]: expected "status(Status)"
      and we read Status from the answer sets.

    Requires python 'clingo' package.
    """
    if not HAS_CLINGO:
        return "asp_backend_missing"

    asp_code = spec["program"]
    status_atom_functor = "status"
    status_value = "unknown"

    ctl = clingo.Control()
    ctl.add("base", [], asp_code)
    ctl.ground([("base", [])])

    def on_model(model: clingo.Model):
        nonlocal status_value
        for s in model.symbols(atoms=True):
            if s.name == status_atom_functor and len(s.arguments) == 1:
                arg = s.arguments[0]
                if arg.type == clingo.SymbolType.Function:
                    status_value = arg.name
                elif arg.type == clingo.SymbolType.String:
                    status_value = arg.string

    ctl.solve(on_model=on_model)

    status_value = status_value.lower()
    if status_value not in {"supported", "contradicted", "unknown"}:
        return "unknown"
    return status_value


# ============================================
# 7. Orchestrator
# ============================================

def validate_with_logic(query: str) -> Dict[str, Any]:
    """
    End-to-end pipeline:
      1. Get LLM answer
      2. Ask LLM to build logical validator (formalism + program + goal)
      3. Run the appropriate backend (Prolog or ASP)
      4. Produce verdict + metadata
    """

    # 1) LLM answer
    answer = get_llm_answer(query)

    # 2) Logical spec (formalism + backend + program)
    logic_spec = get_logic_spec(query, answer)
    formalism = logic_spec["formalism"]
    backend = logic_spec.get("backend", "prolog")

    # 3) Execute backend
    if backend == "prolog":
        verdict = run_prolog_backend(logic_spec)
    elif backend == "asp":
        verdict = run_asp_backend(logic_spec)
    else:
        verdict = "unsupported_backend"

    supported = verdict == "supported"
    contradicted = verdict == "contradicted"

    cfg = FORMALISM_CONFIGS.get(formalism)
    python_deps = cfg.python_deps if cfg else []

    return {
        "query": query,
        "llm_answer": answer,
        "logic_formalism": formalism,
        "logic_formalism_human": cfg.name if cfg else None,
        "backend": backend,
        "python_dependencies": python_deps,
        "prolog_or_asp_notes": cfg.notes if cfg else "",
        "verdict": verdict,
        "is_supported": supported,
        "is_contradicted": contradicted,
        "logic_spec": logic_spec,  # full spec for logging / debugging
    }


# ============================================
# 8. Example usage
# ============================================

if __name__ == "__main__":
    example_query = "Can walking in cold sea water directly cause gout in a healthy adult?"
    result = validate_with_logic(example_query)

    print("=== Logical Validation Result ===")
    print("Formalism:         ", result["logic_formalism"], " – ", result["logic_formalism_human"])
    print("Backend:           ", result["backend"])
    print("Python deps:       ", result["python_dependencies"])
    print("Notes:             ", result["prolog_or_asp_notes"])
    print("LLM answer:        ", result["llm_answer"])
    print("Verdict:           ", result["verdict"])
    print("Supported?         ", result["is_supported"])
    print("Contradicted?      ", result["is_contradicted"])
