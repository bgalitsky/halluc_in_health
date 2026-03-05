import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ----------------------------
# Controlled relation inventory
# ----------------------------
RELATIONS = [
    "CAUSE", "RESULT", "CONDITION", "PURPOSE", "CONTRAST", "CONCESSION",
    "ELABORATION", "BACKGROUND", "EVIDENCE", "JUSTIFICATION", "TEMPORAL",
    "ENABLEMENT", "SUMMARY", "JOINT"
]

DOMAINS = ["politics", "news_business", "engineering", "legal", "health", "security", "science", "finance"]

# ----------------------------
# Paragraph templates (each yields 4-7 EDUs)
# You can add more templates to increase diversity.
# ----------------------------
TEMPLATES = {
    "politics": [
        ("Although {actor} denied {claim}, {org} reopened {topic} after {evidence}. "
         "{org} said {detail}, and it requested {action} to verify {goal}."),
        ("While {policy} was framed as a compromise, critics warned {risk}. "
         "Because {trigger}, the coalition demanded {demand}, but {actor} insisted {stance}.")
    ],
    "news_business": [
        ("The firm revised its forecast, which investors read as {signal}. "
         "As a result, shares fell during {time}, even though {counterfact}. "
         "Analysts said {rationale}."),
        ("After {event}, traders rotated into {asset}, so volatility eased. "
         "However, liquidity remained thin because {cause}, and spreads widened.")
    ],
    "engineering": [
        ("If {cond}, {effect1} and {effect2}. "
         "Therefore {response}, but {side_effect}. "
         "To {purpose}, {operator_action}."),
        ("Because {cause}, the reactor temperature drifted upward, which reduced selectivity. "
         "Engineers added {fix} to stabilize the profile, and they updated {control} to prevent recurrence.")
    ],
    "legal": [
        ("Because the contract defines {term}, the supplier remains liable for {liability} even if {exception}. "
         "The court noted {note}, and it required {requirement} before granting {relief}."),
        ("Although the judge issued {order}, the ruling emphasized {condition}. "
         "Since {basis}, the parties must {obligation}, but either side may {option}.")
    ],
    "health": [
        ("While {metric} improved after {treatment}, {symptom} suggests {assessment}. "
         "Because {reason}, clinicians recommended {plan}, and they warned {caution}."),
        ("If {trigger}, {outcome} is more likely, so the patient should {advice}. "
         "However, {constraint} limits options, and follow-up is needed to confirm {target}.")
    ],
    "security": [
        ("After the breach was detected, responders isolated {system} and revoked {credential}. "
         "Although logs were incomplete, indicators suggested {ioc}, so the team prioritized {priority}."),
        ("Because {vuln} remained unpatched, attackers escalated privileges, which enabled {impact}. "
         "To reduce exposure, the org deployed {mitigation}, but {tradeoff} slowed operations.")
    ],
    "science": [
        ("Although the first experiment appeared conclusive, replication failed when {variation}. "
         "Because {explanation}, the authors revised {model}, and they preregistered {study}."),
        ("If {assumption} holds, the data imply {implication}. "
         "Therefore the team collected {data}, but uncertainty remains due to {uncertainty}.")
    ],
    "finance": [
        ("Since inflation eased, yields declined, so borrowers refinanced {instrument}. "
         "However, credit spreads stayed wide because {risk}, and investors demanded {premium}."),
        ("Although the portfolio was diversified, correlation spiked during {shock}. "
         "As a result, hedges underperformed, and risk limits were tightened to prevent {failure}.")
    ]
}

# ----------------------------
# Slot dictionaries for each domain
# ----------------------------
SLOTS = {
    "politics": {
        "actor": ["the minister", "the governor", "the opposition leader", "the spokesperson"],
        "claim": ["wrongdoing", "involvement", "conflict-of-interest allegations", "abuse of office"],
        "org": ["the parliamentary committee", "the ethics commission", "the oversight board"],
        "topic": ["the inquiry", "the investigation", "the procurement review"],
        "evidence": ["leaked messages hinted at a conflict of interest", "an internal memo contradicted the public statement", "witness testimony emerged"],
        "detail": ["the timeline appeared inconsistent", "the disclosure was incomplete", "the procurement process lacked documentation"],
        "action": ["phone records", "meeting minutes", "email archives"],
        "goal": ["the chronology", "the decision path", "the funding trail"],
        "policy": ["the budget amendment", "the migration proposal", "the public safety bill"],
        "risk": ["it could erode accountability", "it would create loopholes", "it might undermine enforcement"],
        "trigger": ["public pressure intensified", "polls shifted sharply", "regional leaders threatened to defect"],
        "demand": ["a formal audit", "a revised timetable", "new disclosure requirements"],
        "stance": ["the process was lawful", "the claims were politicized", "the vote should proceed"]
    },
    "news_business": {
        "signal": ["a demand slowdown", "margin compression", "weaker guidance", "inventory overhang"],
        "time": ["premarket trading", "the opening auction", "a late-session selloff"],
        "counterfact": ["revenue hit a record last quarter", "shipments rose year-over-year", "the firm beat EPS estimates"],
        "rationale": ["guidance matters more than the headline numbers", "forward bookings softened", "unit economics deteriorated"],
        "event": ["the central bank announcement", "an earnings surprise", "a geopolitical headline"],
        "asset": ["defensives", "short-duration bonds", "cash equivalents", "quality equities"],
        "cause": ["dealers reduced balance sheet", "macro uncertainty persisted", "positioning was crowded"],
    },
    "engineering": {
        "cond": ["fouling doubles in the heat exchanger", "the feed temperature rises by 20°C", "the catalyst activity drops by 15%"],
        "effect1": ["the outlet temperature drops", "the conversion falls", "the pressure profile shifts"],
        "effect2": ["the duty falls below target", "the product spec is missed", "the compressor trips more often"],
        "response": ["the controller increases flow to restore performance", "operators adjust the bypass valve", "the control system tightens setpoints"],
        "side_effect": ["the higher velocity raises pressure drop", "energy consumption increases", "vibration risk rises"],
        "purpose": ["limit pumping costs", "avoid off-spec production", "reduce thermal stress"],
        "operator_action": ["the operator schedules cleaning before the next campaign", "maintenance replaces the filter element", "engineering recalibrates sensors"],
        "cause": ["a cooling loop degraded", "instrument drift accumulated", "ambient conditions changed"],
        "fix": ["a heat sink", "a tuned damper", "an additional recirculation line"],
        "control": ["the alarm thresholds", "the MPC constraints", "the interlock logic"],
    },
    "legal": {
        "term": ["\"delivery\" as the moment goods reach the buyer’s warehouse", "\"acceptance\" as written confirmation", "\"confidential information\" broadly"],
        "liability": ["damage in transit", "latent defects", "delay penalties"],
        "exception": ["the carrier is subcontracted", "the buyer arranged pickup", "the goods are stored temporarily"],
        "note": ["the parties allocated risk explicitly", "the record showed notice was given", "the clause was unambiguous"],
        "requirement": ["a bond", "a verified declaration", "an escrow"],
        "relief": ["injunctive relief", "specific performance", "expedited discovery"],
        "order": ["a temporary injunction", "a protective order", "summary judgment"],
        "condition": ["the plaintiff must show irreparable harm", "the remedy is narrowly tailored", "compliance is subject to review"],
        "basis": ["the statute imposes strict notice deadlines", "the forum clause is enforceable", "precedent controls the standard"],
        "obligation": ["produce documents within 14 days", "preserve evidence", "enter mediation"],
        "option": ["seek interlocutory appeal", "request modification", "move to compel arbitration"]
    },
    "health": {
        "metric": ["CRP", "ESR", "pain score", "blood pressure"],
        "treatment": ["steroids were started", "colchicine was initiated", "the dose was adjusted"],
        "symptom": ["persistent morning stiffness", "recurrent swelling", "ongoing fatigue"],
        "assessment": ["inflammation is not fully controlled", "another diagnosis is possible", "the flare is only partially resolved"],
        "reason": ["infection must be excluded", "renal function limits NSAID use", "symptoms are atypical"],
        "plan": ["a follow-up panel", "imaging", "a taper schedule"],
        "caution": ["to monitor for side effects", "to avoid dehydration", "to seek care if fever develops"],
        "trigger": ["dehydration occurs", "a high-purine meal happens", "sleep deprivation persists"],
        "outcome": ["a flare", "arrhythmia symptoms", "blood pressure spikes"],
        "advice": ["increase fluids and rest", "avoid alcohol", "check temperature twice daily"],
        "constraint": ["one kidney", "anticoagulation therapy", "drug interactions"]
    },
    "security": {
        "system": ["the mail server", "the bastion host", "the database cluster"],
        "credential": ["API tokens", "VPN certificates", "privileged keys"],
        "ioc": ["credential stuffing", "malicious PowerShell", "lateral movement via SMB"],
        "priority": ["account resets", "network segmentation", "forensic imaging"],
        "vuln": ["an outdated OpenSSL package", "a weak IAM policy", "a missing patch"],
        "impact": ["data exfiltration", "ransomware deployment", "service disruption"],
        "mitigation": ["MFA everywhere", "WAF rules", "least-privilege roles"],
        "tradeoff": ["deployment latency", "false positives", "higher operational overhead"]
    },
    "science": {
        "variation": ["the protocol changed slightly", "the sample size was doubled", "a different instrument was used"],
        "explanation": ["measurement bias was likely", "the effect depends on context", "confounders were underestimated"],
        "model": ["the causal diagram", "the prior assumptions", "the statistical specification"],
        "study": ["a replication study", "a blinded analysis", "a multi-site trial"],
        "assumption": ["linearity", "stationarity", "independence"],
        "implication": ["a stronger effect", "no meaningful difference", "a phase transition"],
        "data": ["additional controls", "higher-resolution measurements", "out-of-domain samples"],
        "uncertainty": ["instrument drift", "selection bias", "unmodeled noise"]
    },
    "finance": {
        "instrument": ["mortgages", "corporate loans", "auto loans"],
        "risk": ["default risk", "geopolitical uncertainty", "earnings volatility"],
        "premium": ["higher coupons", "tighter covenants", "more collateral"],
        "shock": ["a liquidity crunch", "a sudden rate spike", "a commodity collapse"],
        "failure": ["excess drawdowns", "concentration breaches", "margin calls"]
    }
}

def fill_template(domain: str, template: str, rng: random.Random) -> str:
    slots = SLOTS[domain]
    out = template
    for key, vals in slots.items():
        token = "{" + key + "}"
        if token in out:
            out = out.replace(token, rng.choice(vals))
    return out

def segment_edus(paragraph: str) -> List[Dict]:
    # Simple EDU segmentation: split on sentence boundaries and coordinating conjunction patterns.
    # Deterministic and stable for synthetic dataset.
    # You can replace with a real EDU segmenter later.
    edus = []
    start = 0
    # naive split points
    split_chars = []
    for i, ch in enumerate(paragraph):
        if ch in ".;":
            split_chars.append(i+1)
    # also split on ", and " / ", but " / ", so "
    for marker in [", and ", ", but ", ", so ", ", therefore ", ", however ", ", although ", ", while ", ", because ", ", since ", ", after ", ", if ", ", even though "]:
        idx = 0
        while True:
            j = paragraph.find(marker, idx)
            if j == -1:
                break
            # split before marker (keep marker with next EDU)
            split_chars.append(j)
            idx = j + len(marker)
    split_chars = sorted(set([p for p in split_chars if 0 < p < len(paragraph)]))
    # build spans
    prev = 0
    edu_id = 1
    for p in split_chars + [len(paragraph)]:
        chunk = paragraph[prev:p].strip()
        if chunk:
            # find exact offsets of chunk in original range
            # We'll approximate by searching from prev
            s = paragraph.find(chunk, prev)
            e = s + len(chunk)
            edus.append({"edu_id": f"e{edu_id}", "text": chunk, "start": s, "end": e})
            edu_id += 1
        prev = p
    return edus

def build_simple_tree(edus: List[Dict], rng: random.Random) -> Dict:
    nodes = []
    node_text = {}   # <-- NEW: stores representative text for any node_id
    node_span = {}   # <-- NEW: stores span for any node_id

    # 1) Create leaf nodes
    leaf_ids = []
    for i, edu in enumerate(edus, start=1):
        lid = f"l{i}"
        leaf_ids.append(lid)
        span = {"start": edu["start"], "end": edu["end"]}
        nodes.append({
            "node_id": lid,
            "type": "leaf",
            "edu_id": edu["edu_id"],
            "span": span
        })
        node_text[lid] = edu["text"]
        node_span[lid] = span

    # Edge case: if EDU segmentation produced 0 EDUs (shouldn't, but safe)
    if not leaf_ids:
        return {"root_id": None, "nodes": []}

    # 2) Relation chooser by discourse cues
    def relation_for(text: str) -> str:
        t = text.lower().strip()
        if t.startswith("if ") or " unless " in t:
            return "CONDITION"
        if t.startswith("because") or t.startswith("since ") or " because " in t or " since " in t:
            return "CAUSE"
        if t.startswith("to ") or " to " in t:
            return "PURPOSE"
        if t.startswith("although") or " even though " in t:
            return "CONCESSION"
        if " but " in t or t.startswith("however") or " however " in t:
            return "CONTRAST"
        if t.startswith("after ") or " after " in t:
            return "TEMPORAL"
        return rng.choice(["ELABORATION", "BACKGROUND", "EVIDENCE", "JUSTIFICATION", "JOINT"])

    def is_satellite_like(text: str) -> bool:
        t = text.lower().lstrip()
        return t.startswith(("although", "because", "since", "if", "to", "after", "however"))

    # 3) Bottom-up binary merge (sequential pairing)
    current = leaf_ids[:]
    internal_idx = 1

    while len(current) > 1:
        left = current.pop(0)
        right = current.pop(0)

        left_span = node_span[left]
        right_span = node_span[right]
        span = {
            "start": min(left_span["start"], right_span["start"]),
            "end": max(left_span["end"], right_span["end"])
        }

        left_text = node_text[left]
        right_text = node_text[right]

        rel = relation_for(left_text + " " + right_text)

        left_sat = is_satellite_like(left_text)
        right_sat = is_satellite_like(right_text)

        # nucleus/satellite heuristic
        if left_sat and not right_sat:
            nucleus, satellite = right, left
        elif right_sat and not left_sat:
            nucleus, satellite = left, right
        else:
            nucleus, satellite = left, right

        nid = f"n{internal_idx}"
        internal_idx += 1

        nodes.append({
            "node_id": nid,
            "type": "internal",
            "relation": rel,
            "nucleus": nucleus,
            "satellite": satellite,
            "children": [left, right],
            "span": span
        })

        # Representative text for internal node: take nucleus text (stable + short)
        node_text[nid] = node_text[nucleus]
        node_span[nid] = span

        # push merged node back
        current.insert(0, nid)

    return {"root_id": current[0], "nodes": nodes}

def generate_dataset(n_rows: int = 1000, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    data = []
    for i in range(1, n_rows + 1):
        domain = rng.choice(DOMAINS)
        template = rng.choice(TEMPLATES[domain])
        paragraph = fill_template(domain, template, rng)
        edus = segment_edus(paragraph)
        tree = build_simple_tree(edus, rng)
        data.append({
            "id": f"row_{i:04d}",
            "domain": domain,
            "paragraph": paragraph,
            "edus": edus,
            "tree": tree
        })
    return data

if __name__ == "__main__":
    dataset = generate_dataset(n_rows=1000, seed=2026)
    with open("rst_paragraph_dataset_1000.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print("Wrote rst_paragraph_dataset_1000.json with", len(dataset), "rows")