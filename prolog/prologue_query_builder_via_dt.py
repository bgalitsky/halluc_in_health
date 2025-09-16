import requests
import json

# Fixed ontology: map EDU text keywords → Prolog predicates
ONTOLOGY = {
    "pain": "symptom(joint_pain).",
    "swollen": "symptom(swelling).",
    "red": "symptom(redness).",
    "warm": "symptom(warmth).",
    "stiff": "symptom(stiffness).",
    "throbbing": "symptom(throbbing_pain).",
    "meal": "symptom(pain_after_meal).",
    "previous gout": "symptom(previous_gout).",
    "past attacks": "symptom(recognized_symptoms)."
}

def edu_to_prolog(edu_text: str) -> list:
    """Convert EDU text into list of Prolog facts based on ontology."""
    facts = []
    text_lower = edu_text.lower()
    for keyword, fact in ONTOLOGY.items():
        if keyword in text_lower:
            facts.append(fact)
    return facts

def analyze_text(text: str):
    """Send text to FastAPI discourse parser, get JSON, and convert EDUs to Prolog clauses."""
    url = "http://54.82.56.2:8000/analyze"
    headers = {"Content-Type": "application/json"}
    payload = {"text": text}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    prolog_facts = []

    def traverse_tree(node):
        # Process current EDU
        if "edu" in node and node["edu"]:
            prolog_facts.extend(edu_to_prolog(node["edu"]))
        # Traverse satellites
        if "satellites" in node:
            for sat in node["satellites"]:
                traverse_tree(sat)

    traverse_tree(data["tree"])

    return prolog_facts


if __name__ == "__main__":
    # Example input text
    text = "For the past two days, my big toe has been extremely painful, swollen, and red. I have had gout before."
    facts = analyze_text(text)

    print("Generated Prolog facts:")
    for f in facts:
        print(f)
