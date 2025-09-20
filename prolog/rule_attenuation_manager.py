import requests
from itertools import chain, combinations
from pyswip import Prolog


class DiscourseParser:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def parse(self, text):
        """Send text to discourse parser API and return JSON parse."""
        payload = {"text": text}
        response = requests.post(self.endpoint, json=payload)
        response.raise_for_status()
        return response.json()

    def map_to_atoms(self, parse_result):
        """
        Map EDUs to logical atoms c1, c2, ...
        Returns dict {cX: EDU_text}, plus nucleus/satellites lists in atom form.
        """
        atoms = {}
        satellites_atoms = []

        # Collect all discourse units
        all_edus = [parse_result["tree"]["nucleus"]] + parse_result.get("satellites_only_with_nucleus", [])
        for i, edu in enumerate(all_edus, start=1):
            atom = f"c{i}"
            atoms[atom] = edu

        # Mark satellites
        for i, edu in enumerate(parse_result.get("satellites_only_with_nucleus", []), start=2):
            satellites_atoms.append(f"c{i}")

        return atoms, satellites_atoms


class DiseaseReasoner:
    def __init__(self, ontology_str):
        self.prolog = Prolog()
        self._load_ontology(ontology_str)

    def _load_ontology(self, ontology_str):
        """Load ontology rules from a string into Prolog."""
        for line in ontology_str.splitlines():
            line = line.strip()
            if line and not line.startswith("%"):  # skip empty or commented lines
                self.prolog.assertz(line)

    def assert_patient_facts(self, facts):
        """Assert patient-specific complaint facts into Prolog."""
        for f in facts:
            self.prolog.assertz(f)

    def check_disease(self, disease_name="gout"):
        """Check if disease can be derived."""
        return bool(list(self.prolog.query(f"disease({disease_name})")))


class DiscourseAttenuator:
    def __init__(self, prolog):
        self.prolog = prolog

    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def attenuate_clause(self, head, body, satellites):
        results = []

        for removal in self.powerset(satellites):
            kept = [c for c in body if c not in removal]
            if not kept:
                continue

            body_str = ", ".join([f for f in kept])
            rule = f"{head} :- {body_str}."
            self.prolog.assertz(rule)

            success = bool(list(self.prolog.query(head)))
            results.append({
                "removed": removal,
                "rule": rule,
                "succeeds": success,
                "kept_count": len(kept)
            })

        successful = [r for r in results if r["succeeds"]]
        best = None
        if successful:
            best = sorted(successful, key=lambda r: (len(r["removed"]), -r["kept_count"]))[0]

        return results, best


# ---------------- Example usage ----------------
if __name__ == "__main__":
    ontology = """
    inflammation(joints(A)) :- joints(A), member(A, [one,few,both,multiple,toe,knee,ankle]).
    inflammation(pain(S)) :- pain(S), member(S, [painfull,severe,throbbing,crushing,excruciating]).
    inflammation(property(C)) :- property(C), member(C, [red,warm,tender,swollen,fever]).
    inflammation(last(L)) :- last(L), member(L, [few_days,return,additional_longer]).
    disease(gout) :- inflammation(joints(A)), inflammation(pain(S)), inflammation(property(C)), inflammation(last(L)).
    """

    # Initialize reasoner with ontology
    reasoner = DiseaseReasoner(ontology)

    # Example patient complaint
    patient_facts = [
        "joints(toe)",
        "pain(painfull)",
        "property(red)",
        "property(warm)",
        "last(few_days)"
    ]
    reasoner.assert_patient_facts(patient_facts)

    print("Does the patient have gout?", reasoner.check_disease("gout"))
