import re


def _split_top_level_commas(s: str) -> list[str]:
    """Split string s on commas that are not inside parentheses or brackets."""
    parts, buf = [], []
    paren, bracket = 0, 0
    for ch in s:
        if ch == '(':
            paren += 1
        elif ch == ')':
            paren = max(0, paren - 1)
        elif ch == '[':
            bracket += 1
        elif ch == ']':
            bracket = max(0, bracket - 1)

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


def _find_member_lists(body_str: str) -> list[str]:
    """
    Find all member(Var, [ ... ]) lists in body_str and return
    each list's inner content string (without the surrounding brackets).
    """
    lists = []
    i = 0
    while True:
        m_idx = body_str.find('member(', i)
        if m_idx == -1:
            break
        # find the first '[' after this member(
        lb = body_str.find('[', m_idx)
        if lb == -1:
            i = m_idx + 7
            continue
        # parse until matching ']' at top-level bracket depth
        depth = 1
        j = lb + 1
        while j < len(body_str) and depth > 0:
            if body_str[j] == '[':
                depth += 1
            elif body_str[j] == ']':
                depth -= 1
            j += 1
        if depth == 0:
            inner = body_str[lb+1:j-1].strip()
            lists.append(inner)
            i = j
        else:
            # unmatched bracket; bail out of this occurrence
            i = lb + 1
    return lists


def extract_ontology(ontology_str: str) -> list[dict]:
    """
    Parse Prolog-style ontology into a list of rules:
      [
        {"head": "predicate(args)", "body": ["g1(...)", "g2(...)", ...], "values": ["v1","v2", ...]},
        ...
      ]
    - Works with nested parentheses in heads/arguments.
    - Splits bodies on commas at top level only.
    - Extracts all member/2 lists' values (flattened) per rule.
    """
    rules = []
    for raw in ontology_str.splitlines():
        line = raw.strip()
        if not line or line.startswith('%'):
            continue
        # Remove trailing period if present
        if line.endswith('.'):
            line = line[:-1].strip()

        if ':-' in line:
            # Head is everything before ':-', body after
            head, body_str = line.split(':-', 1)
            head = head.strip()
            body_str = body_str.strip()
            body = _split_top_level_commas(body_str)

            # Collect all value atoms appearing in any member/2 list in this body
            values = []
            for inner in _find_member_lists(body_str):
                vals = _split_top_level_commas(inner)
                values.extend([v.strip() for v in vals if v.strip()])
            rules.append({"head": head, "body": body, "values": values})
        else:
            # fact
            rules.append({"head": line, "body": [], "values": []})
    return rules


# --------- Optional: mapping satellites to clause atoms using the parsed ontology ---------

def _norm(txt: str) -> str:
    return txt.lower().replace(' ', '_')

def map_satellites_to_atoms(goal_clause: str, discourse: dict, ontology_str: str) -> dict:
    """
    Return:
      {
        "clause_body": [subgoal,...],
        "satellite_map": { satellite_text: matching_body_atom, ... }
      }
    Matching is done by checking whether any member/2 value (normalized) appears in the
    normalized satellite text, and if that value belongs to the rule whose HEAD equals
    a subgoal present in the goal clause body.
    """
    rules = extract_ontology(ontology_str)
    # Index by head for quic



def normalize(text: str) -> str:
    return text.lower().replace(" ", "_")


def map_satellites_to_atoms(goal_clause: str, discourse: dict, ontology_str: str):
    """
    Combine ontology parsing, goal clause parsing, and discourse mapping.
    Returns clause body and satellite→atom mapping.
    """
    rules = extract_ontology(ontology_str)

    # Index ontology by head
    ontology = {r["head"]: r for r in rules}

    # Extract atoms from goal clause
    body_match = re.search(r":-(.*)", goal_clause)
    if not body_match:
        return {"clause_body": [], "satellite_map": {}}
    clause_body = [a.strip() for a in body_match.group(1).split(",")]

    # Collect satellites from discourse
    satellites_text_list = discourse.get("dependent_satellites", [])
    satellite_map = {}

    for sat in satellites_text_list:
        sat_norm = normalize(sat)
        for atom, rule in ontology.items():
            if any(v in sat_norm for v in rule["values"]):
                satellite_map[sat] = atom

    return {"clause_body": clause_body, "satellite_map": satellite_map}


# ---------------- Main script ----------------
if __name__ == "__main__":
    ontology_str = """
    inflammation(joints(A)) :- joints(A), member(A,[one,few,both,multiple,toe,knee,ankle]).
    inflammation(pain(S)) :- pain(S), member(S,[painfull,severe,throbbing,crushing,excruciating]).
    inflammation(property(C)) :- property(C), member(C,[red,warm,tender,swollen,fever]).
    inflammation(last(L)) :- last(L), member(L,[few_days,return,additional(longer)]).
    """

    goal_clause = "disease(gout) :- inflammation(joints(A)), inflammation(pain(S)), inflammation(property(C)), inflammation(last(L))"

    discourse = {
        "dependent_satellites": [
            "For the past few days",
            "in my knee and ankle",
            "pain was throbbing and severe",
            "the joint looks red",
            "and is swollen",
            "I also had a fever"
        ],
        "tree": {
            "edu": None,
            "nucleus": {
                "edu": "I could barely walk",
                "nucleus": None,
                "relation": None,
                "satellites": [
                    {"edu": "pain was throbbing and severe", "relation": "elaboration", "satellites": []},
                    {"edu": "the joint looks red", "relation": "joint", "satellites": []},
                    {"edu": "and is swollen", "relation": "joint", "satellites": []}
                ]
            },
            "relation": None,
            "satellites": [
                {"edu": "For the past few days", "relation": "temporal", "satellites": []},
                {"edu": "in my knee and ankle", "relation": "location", "satellites": []},
                {"edu": "I also had a fever", "relation": "background", "satellites": []}
            ]
        }
    }

    sats = map_satellites_to_atoms(goal_clause, discourse, ontology_str)
    print("Satellite atoms:", sats)
