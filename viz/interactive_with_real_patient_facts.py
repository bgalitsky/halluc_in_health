from typing import List, Dict, Tuple
from itertools import chain, combinations
from pyswip import Prolog

# ---------- Small helpers ----------

def powerset(iterable):
    """All non-empty subsets of an iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def strip_trailing_period(clause: str) -> str:
    clause = clause.strip()
    return clause[:-1].strip() if clause.endswith('.') else clause

def safe_assert_facts(prolog: Prolog, facts: List[str]):
    """Assert facts; skip any malformed ones."""
    for f in facts:
        try:
            prolog.assertz(strip_trailing_period(f))
        except Exception as e:
            print(f"⚠️ Skipping fact '{f}': {e}")

# ---------- Ontology parsing (generic) ----------

def _split_top_level_commas(s: str) -> List[str]:
    parts, buf, paren, bracket = [], [], 0, 0
    for ch in s:
        if ch == '(':
            paren += 1
        elif ch == ')':
            paren -= 1
        elif ch == '[':
            bracket += 1
        elif ch == ']':
            bracket -= 1
        if ch == ',' and paren == 0 and bracket == 0:
            part = ''.join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = ''.join(buf).strip()
    if tail:
        parts.append(tail)
    return parts

def extract_rules(ontology_str: str) -> List[Dict]:
    """
    Return list of rules/facts: [{"head": "...", "body": [...]}]
    Works for any predicates; no assumptions about names.
    """
    rules = []
    for raw in ontology_str.splitlines():
        line = strip_trailing_period(raw.strip())
        if not line or line.startswith("%"):
            continue
        if ":-" in line:
            head, body_str = line.split(":-", 1)
            head = head.strip()
            body = _split_top_level_commas(body_str.strip())
            rules.append({"head": head, "body": body})
        else:
            rules.append({"head": line, "body": []})
    return rules

def find_goal_rules(ontology_str: str, goal_predicate_prefix: str = "disease(") -> List[Dict]:
    """
    Heuristic: treat any rule whose head starts with goal_predicate_prefix as a goal rule.
    (You can change prefix to 'repair(' or 'recommend(' for other domains.)
    """
    return [r for r in extract_rules(ontology_str) if r["head"].startswith(goal_predicate_prefix)]

# ---------- Attenuation core ----------

def attenuate_one_goal(
    prolog: Prolog,
    head: str,
    body_atoms: List[str],
    removable_atoms: List[str]
) -> Tuple[List[Dict], Dict]:
    """
    Test the full rule + all attenuated versions (remove subsets of removable_atoms).
    Returns (results, best): best = minimal removal, ties → more kept.
    """
    results = []

    # Test full rule first
    full_rule = f"{head} :- {', '.join(body_atoms)}" if body_atoms else head
    success_full = False
    try:
        prolog.assertz(full_rule)
        success_full = bool(list(prolog.query(head)))
    except Exception as e:
        print(f"⚠️ Full rule failed to run: {e}")
    finally:
        try:
            prolog.retract(full_rule)
        except Exception:
            pass
    results.append({"removed": (), "rule": full_rule, "succeeds": success_full, "kept_count": len(body_atoms)})

    # Try attenuations (non-empty removals)
    for removal in powerset(removable_atoms):
        kept = [a for a in body_atoms if a not in removal]
        if not kept:  # must keep at least one atom
            continue

        rule = f"{head} :- {', '.join(kept)}"
        success = False
        try:
            prolog.assertz(rule)
            success = bool(list(prolog.query(head)))
        except Exception as e:
            # keep going even if this attempt fails
            print(f"⚠️ Attenuated rule failed: {rule} → {e}")
        finally:
            try:
                prolog.retract(rule)
            except Exception:
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

# ---------- Multi-disease runner ----------

class MultiGoalAttenuator:
    """
    Run attenuation for all goals in an ontology (default prefix = 'disease('),
    using a single set of patient facts.
    """
    def __init__(self, ontology_str: str, goal_predicate_prefix: str = "disease("):
        self.ontology_str = ontology_str
        self.goal_predicate_prefix = goal_predicate_prefix

    def run(
        self,
        patient_facts: List[str],
        removable_by_goal: Dict[str, List[str]] = None
    ) -> Dict[str, Dict]:
        """
        For each goal rule head in the ontology:
          - Asserts ontology + facts
          - Runs attenuation with atoms marked removable for that goal (if provided)
            else defaults to "all atoms in the body are removable".
        Returns {goal_head: {"results": [...], "best": {...}}}
        """
        # Parse ontology & pick goal rules
        all_rules = extract_rules(self.ontology_str)
        goal_rules = [r for r in all_rules if r["head"].startswith(self.goal_predicate_prefix)]
        if not goal_rules:
            print("⚠️ No goal rules found with prefix", self.goal_predicate_prefix)
            return {}

        out = {}
        for goal_rule in goal_rules:
            head = goal_rule["head"]
            body = goal_rule["body"]

            # Build a fresh Prolog engine per goal to avoid state bleed
            prolog = Prolog()

            # Assert all ontology rules/facts
            for r in all_rules:
                clause = r["head"] if not r["body"] else f"{r['head']} :- {', '.join(r['body'])}"
                try:
                    prolog.assertz(clause)
                except Exception as e:
                    print(f"⚠️ Skipping bad clause: {clause} → {e}")

            # Assert patient facts
            safe_assert_facts(prolog, patient_facts)

            # Decide which atoms are removable for this goal
            removable = (removable_by_goal or {}).get(head, body[:])  # default: all body atoms
            results, best = attenuate_one_goal(prolog, head, body, removable)

            out[head] = {"results": results, "best": best}

        return out

# ---------- (Optional) Visualization per goal ----------

def draw_attenuation_tree_for_goal(goal_head: str, body_atoms: List[str], removable_atoms: List[str], results: List[Dict]):
    """
    Color nodes by success/failure based on results.
    Requires networkx & matplotlib (already used above).
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    # Build success set from results
    success_nodes = set()
    node_labels = {}

    root = goal_head
    node_labels[root] = root
    for r in results:
        if r["removed"]:
            node = f"{goal_head} - {list(r['removed'])}"
        else:
            node = goal_head
        node_labels[node] = node
        if r["succeeds"]:
            success_nodes.add(node)

    # Build tree edges (simple star from root to each attenuation)
    G = nx.DiGraph()
    G.add_node(root)
    for r in results:
        if r["removed"]:
            node = f"{goal_head} - {list(r['removed'])}"
            G.add_edge(root, node)

    colors = ["lightgreen" if n in success_nodes else "salmon" for n in G.nodes()]

    plt.figure(figsize=(14, 8))
    pos = nx.spring_layout(G, seed=42, k=0.7)
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=3500,
            node_color=colors, font_size=8, font_weight="bold",
            edge_color="gray", arrows=True)
    plt.title(f"Attenuation Tree: {goal_head}\nGreen = Success, Red = Fail", fontsize=14, weight="bold")
    plt.show()

# Example ontology with two diseases
ontology = """
% support rules (toy)
pain(knee). swelling(knee). stiffness(morning).
symptom(fatigue). inflammation(systemic). involvement(heart).
pain(big_toe). swelling(big_toe). redness(big_toe).
uric_acid(high). onset(sudden).

% disease rules
disease(arthritis) :-
    pain(J), swelling(J), stiffness(morning),
    symptom(fatigue), inflammation(systemic), involvement(O).

disease(gout) :-
    pain(big_toe), swelling(big_toe), redness(big_toe),
    uric_acid(high), onset(sudden)).
"""

# Patient facts (could be a subset)
patient_facts = [
    "pain(knee)",
    "swelling(knee)",
    "stiffness(morning)",
    "inflammation(systemic)",
    # intentionally missing fatigue & involvement/organ
    "pain(big_toe)",
    "swelling(big_toe)",
    "redness(big_toe)",
    "uric_acid(high)",
    "onset(sudden)"
]

# Which atoms can be removed (satellites) for each goal (optional).
# If you omit this, the code treats ALL body atoms as removable by default.
removable_by_goal = {
    "disease(arthritis)": ["symptom(fatigue)", "involvement(O)"],  # keep pain/swelling/stiffness/inflammation
    "disease(gout)": ["redness(big_toe)"]  # e.g., let redness be removable for demo
}

runner = MultiGoalAttenuator(ontology)
all_results = runner.run(patient_facts, removable_by_goal=removable_by_goal)

# Print summary
for goal_head, payload in all_results.items():
    print(f"\n=== {goal_head} ===")
    for r in payload["results"]:
        removed = list(r["removed"]) if r["removed"] else []
        print(f"Removed: {removed:<40} Success: {r['succeeds']}  Rule: {r['rule']}")
    print("Best:", payload["best"])

# Optional: draw trees
# For each goal, pass the body/removable you used, and the results set
# (You can parse bodies again from extract_rules if you prefer.)
for goal_rule in find_goal_rules(ontology):
    head = goal_rule["head"]
    body = goal_rule["body"]
    removable = removable_by_goal.get(head, body[:])
    draw_attenuation_tree_for_goal(head, body, removable, all_results[head]["results"])
