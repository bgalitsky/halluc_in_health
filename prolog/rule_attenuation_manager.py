from doctest import testfile

import requests
from itertools import chain, combinations
from pyswip import Prolog
from functools import lru_cache
from joblib import Memory
import itertools
from satellites_to_atoms_mapper import map_satellites_to_atoms
import re

EC2_PUBLIC_IP = "54.82.56.2"

def is_valid_prolog_clause(f: str) -> bool:
    """
    Validate if a string looks like a Prolog clause.
    Accepts both facts and rules with lists/commas.
    """
    f = f.strip().rstrip(".")

    # Fact: predicate(args)
    fact_pattern = r"^[a-z][a-zA-Z0-9_]*\s*\(.*\)$"

    # Rule: head :- body
    rule_pattern = r"^[a-z][a-zA-Z0-9_]*\s*\(.*\)\s*:-\s*.+$"

    return bool(re.match(fact_pattern, f) or re.match(rule_pattern, f))

def format_reasoning_output(result: dict) -> str:
    """
    Convert the reasoning result dictionary into a pretty string for printing.
    """
    lines = []
    lines.append("=== Reasoning Summary ===")
    lines.append(f"Goal: {result.get('goal')}")
    lines.append(f"Facts: {', '.join(result.get('facts', []))}")
    lines.append(f"Original Check: {result.get('original_check')}")
    lines.append("")

    # Trace
    trace = result.get("trace", {})
    lines.append("Trace:")
    if isinstance(trace, dict):
        for key, val in trace.items():
            lines.append(f"  {key}: {val}")
    else:
        lines.append(f"  {trace}")
    lines.append("")

    # Results
    lines.append("Attenuation Results:")
    for r in result.get("results", []):
        removed = ", ".join(r.get("removed", [])) if r.get("removed") else "None"
        succ = "✔️" if r.get("succeeds") else "❌"
        lines.append(f"  Removed: {removed:<40} | Success: {succ}")

    # Best
    best = result.get("best")
    if best:
        lines.append("")
        lines.append("Best Attenuation:")
        removed = ", ".join(best.get("removed", [])) if best.get("removed") else "None"
        succ = "✔️" if best.get("succeeds") else "❌"
        lines.append(f"  Removed: {removed}")
        lines.append(f"  Rule: {best.get('rule')}")
        lines.append(f"  Success: {succ}")
    else:
        lines.append("")
        lines.append("Best Attenuation: None")

    return "\n".join(lines)

def strip_trailing_period(clause: str) -> str:
    """
        Remove a trailing '.' from a Prolog clause string, if present.
    """
    clause = clause.strip()
    if clause.endswith('.'):
        return clause[:-1].strip()
    return clause

class DiseaseReasoner:
    def __init__(self, ontology_str: str):
        self.prolog = Prolog()
        self._load_ontology(ontology_str)



    def _load_ontology(self, ontology_str: str):
        for line in ontology_str.splitlines():
            line = line.strip()
            if line and not line.startswith("%") and is_valid_prolog_clause(line):
                line_ = strip_trailing_period(line)
                try:
                    self.prolog.assertz(line_)
                except Exception as e:
                    print(f"⚠️ Skipping clause '{line_}': {e}")
            else:
                print(f"Skipping clause {line}")


    def assert_patient_facts(self, facts: list[str]):
        for f in facts:
            f_ = f.strip().rstrip()
            f = strip_trailing_period(f_)
            if is_valid_prolog_clause(f):
                try:
                    self.prolog.assertz(f)
                except Exception as e:
                    print(f"⚠️ Skipping fact '{f}': {e}")
            else:
                print("invalid prolog clause: "+f)

    def check_disease(self, disease_name: str):
        try:
            query = f"disease({disease_name})"
            qres = self.prolog.query(query)
            qres.close()
            return bool(list(qres))
        except Exception as e:
            print(f"⚠️ Prolog query failed: {e}")
            return False


    def trace_inference(self, disease_name: str):
        trace = {}
        try:
            trace["joints"] = list(self.prolog.query("inflammation(joints(A))"))
            trace["pain"] = list(self.prolog.query("inflammation(pain(S))"))
            trace["properties"] = list(self.prolog.query("inflammation(property(C))"))
            trace["last"] = list(self.prolog.query("inflammation(last(L))"))
            trace["disease"] = list(self.prolog.query(f"disease({disease_name})"))
            return trace
        except Exception as e:
            print(f"⚠️ Prolog query failed: {e}")
            return None



# ---------------- Ontology ----------------
ontology = """inflammation(joints(A)) :- joints(A), member(A, [one,few,both,multiple,toe,knee,ankle])
inflammation(pain(S)) :- pain(S), member(S, [painfull,severe,throbbing,crushing,excruciating])
inflammation(property(C)) :- property(C), member(C, [red,warm,tender,swollen,fever])
inflammation(last(L)) :- last(L), member(L, [few_days,return,additional_longer])
disease(gout) :- inflammation(joints(A)), inflammation(pain(S)), inflammation(property(C)), inflammation(last(L))
"""


# ---------------- Call discourse parser ----------------

memory = Memory("./llm_prolog_cache", verbose=1, compress=1)

@lru_cache(maxsize=128)
@memory.cache
def call_discourse_parser(text: str, endpoint: str = f"http://{EC2_PUBLIC_IP}:8000/analyze"):
    response = requests.post(endpoint, json={"text": text})
    response.raise_for_status()
    return response.json()


# ---------------- Clause attenuation ----------------
def powerset(iterable):
    """All subsets of iterable"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def dump_kb(prolog, predicate=None):
    """
    Print all clauses and facts currently asserted in the Prolog engine.
    - If predicate is None, returns everything (`listing/0`).
    - If predicate is provided (e.g. 'disease'), returns just those clauses.
    """
    try:
        if predicate:
            query = f"listing({predicate})"
        else:
            query = "listing"
        results = prolog.query(query)
        results.close()
        return list(results)
    except Exception as e:
        print(f"⚠️ Could not retrieve clauses: {e}")
        return []


def attenuate_disease_clause(prolog, disease_name, body_atoms, satellite_atoms):
    """
    Try attenuating disease clause by removing satellite atoms.
    Ensures each attenuated rule is retracted after testing.
    """
    results = []

    print(" disease_name = "+disease_name +" \n body_atoms = "+ ', '.join(body_atoms) + "\n satellite_atoms = " + ' '.join(satellite_atoms))

    for removal in powerset(satellite_atoms):
        kept = [a for a in body_atoms if a not in removal]
        if not kept:
            continue

        body_str = ", ".join(kept)
        rule = f"{disease_name} :- {body_str}"
        rule_ = strip_trailing_period(rule)
        print("Testing rule:", rule_)

        success = False
        try:
            prolog.assertz(rule_)
            #print('\n'.join(dump_kb(prolog)))

            try:
                qres = prolog.query(f"{disease_name}")
                success = bool(list(qres))
                qres.close()
            except Exception as e:
                print(f"⚠️ Query failed for {disease_name}: {e}")
        except Exception as e:
            print(f"⚠️ Assertion failed for rule {rule_}: {e}")
        finally:
            try:
                prolog.retract(rule_)
            except Exception:
                # Sometimes retract fails if assertion never happened or rule malformed
                print("Problem retracting " + rule_)
                pass

        results.append({
            "removed": removal,
            "rule": rule,
            "succeeds": success,
            "kept_count": len(kept)
        })

    # pick best successful result
    successful = [r for r in results if r["succeeds"]]
    best = None
    if successful:
        best = sorted(successful, key=lambda r: (len(r["removed"]), -r["kept_count"]))[0]

    return results, best



# ---------------- Placeholder: text → facts ----------------
#def text_to_facts(text: str) -> list[str]:
#    # Stub implementation
#    return ["joints(toe)", "pain(painfull)", "property(red)", "property(warm)", "last(few_days)"]

class AttenuatedReasoner:
    def __init__(self, ontology_str: str):
        self.ontology_str = ontology_str

    def run_w_attenuation(self, complaint_text: str, patient_facts: list[str]):
        """
        Run reasoning with discourse-driven attenuation.
        Returns (results, best) attenuation.
        """
        try:
            # get last non-empty line
            last_line = [ln.strip() for ln in self.ontology_str.splitlines() if ln.strip()][-1]
            head = last_line.split(":-", 1)[0].strip()
            #match = re.match(r"^([a-zA-Z0-9_]+)\s*\(", head)
            #if match:
            #    goal =  match.group(1)
            #else:
            #    return None
        except IndexError:
            print("❌ Ontology string is empty.")
            return None

            # check if last line is a goal clause
        if ":-" not in last_line:
            print("❌ Last line in ontology is not a goal clause. Aborting run().")
            return None
        # 1. Call discourse parser
        discourse = call_discourse_parser(complaint_text)
        print("Discourse parser output:", discourse)

        # 2. Find main goal + mapping
        mapping = map_satellites_to_atoms(last_line, discourse, self.ontology_str )
        satellites = list(mapping["satellite_map"].values())
        print("Satellite → Atom mapping:", mapping["satellite_map"])

        # 3. Reasoning
        reaso_ner = DiseaseReasoner(self.ontology_str)
        reaso_ner.assert_patient_facts(patient_facts)

        results, best = attenuate_disease_clause(
            reaso_ner.prolog,
            head,
            #f"disease({goal})",
            mapping["clause_body"],
            satellites
        )
        reaso_ner.prolog.query("halt")
        return {
            "facts": patient_facts,
            "goal": head, #f"disease({goal})",
            "original_check": reaso_ner.check_disease(head),
            "trace": reaso_ner.trace_inference(head),
            "results": results,
            "best": best
        }


# ---------------- Main script ----------------
if __name__ == "__main__":
    #test for main function
    p = Prolog()



    complaint_text = ("For the past few days in my knee and ankle pain was throbbing. When I woke up at night due to severe pain, I took a painkiller. "
                      "When I carefully looked at my red joint, it seemed swollen. Once I discovered that I had a fever, I started thinking how to cure it.")
    # 2. Map text → Prolog facts
    patient_facts = ["joints(toe)", "pain(severe)", "property(red)", "last(few_days)"]

    reasoner = AttenuatedReasoner(ontology)


    reas = DiseaseReasoner(ontology)
    reas.assert_patient_facts(["joint(wrist)."])

    result = reasoner.run_w_attenuation(complaint_text, patient_facts)
    print(result)


    # Another test
    p.assertz("inflammation(joints(A)) :- joints(A), member(A,[toe,knee,ankle])")
    p.assertz("inflammation(pain(S)) :- pain(S), member(S,[painfull,severe])")
    p.assertz("disease(gout) :- inflammation(joints(A)), inflammation(pain(S))")

    p.assertz("joints(toe)")
    p.assertz("pain(painfull)")

    print(list(p.query("disease(gout)")))



    # 1. Call discourse parser
    discourse = call_discourse_parser(complaint_text)
    print("Discourse parser output:", discourse)

    # Use discourse + ontology to find satellite subgoals in the disease clause
    goal_clause = "disease(gout) :- inflammation(joints(A)), inflammation(pain(S)), inflammation(property(C)), inflammation(last(L))"
    mapping = map_satellites_to_atoms(goal_clause, discourse, ontology)
    satellites = list(mapping["satellite_map"].values())
    print("Satellite → Atom mapping:", mapping["satellite_map"])



    # 3. Reasoning
    reasoner = DiseaseReasoner(ontology)
    reasoner.assert_patient_facts(patient_facts)

    # 4. Attenuation of disease(gout) clause
    body_atoms = mapping["clause_body"]

    results, best = attenuate_disease_clause(reasoner.prolog, "gout", body_atoms, satellites)

    print("Attenuation results:")
    for r in results:
        print("Removed:", r["removed"], "Succeeds?", r["succeeds"])
    print("Best attenuation:", best)

    print("\nFacts:", patient_facts)
    print("Original disease check:", reasoner.check_disease("gout"))
    print("Trace:", reasoner.trace_inference("gout"))

    print("\nAttenuation results:")
    for r in results:
        print("Removed:", r["removed"], "Succeeds?", r["succeeds"])
    print("Best attenuation:", best)
    reasoner.prolog.query("halt")
