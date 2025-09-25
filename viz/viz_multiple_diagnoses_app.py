# streamlit_app.py
import streamlit as st
from pyswip import Prolog
from attenuation_engine import MultiGoalAttenuator, find_goal_rules, draw_attenuation_tree_for_goal

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")


# --- Load ontology ---
st.title("Interactive Disease Reasoner with Rule Attenuation")

ontology_str = st.text_area(
    "Paste your ontology here:", """  disease(arthritis) :- pain(J), swelling(J), stiffness(morning), symptom(fatigue), inflammation(systemic), involvement(O).
    disease(gout) :- pain(Joint), swelling(Joint), redness(Joint), uric_acid(high), onset(sudden).
    """
)

# --- Patient facts input ---
patient_facts_text = st.text_area(
    "Enter patient facts (one per line):",
    "pain(knee)\nswelling(knee)\nstiffness(morning)\ninflammation(systemic)\npain(big_toe)\nswelling(big_toe)\nredness(big_toe)\nuric_acid(high)\nonset(sudden)"
)

patient_facts = [line.strip() for line in patient_facts_text.splitlines() if line.strip()]
print(patient_facts)

# --- Run button ---
if st.button("Run Reasoning"):
    runner = MultiGoalAttenuator(ontology_str)
    results = runner.run(patient_facts)

    for goal_rule in find_goal_rules(ontology_str):
        head = goal_rule["head"]
        st.subheader(f"Results for {head}")

        payload = results.get(head)
        if not payload:
            st.write("⚠️ No results.")
            continue

        # Print textual results
        for r in payload["results"]:
            removed = list(r["removed"]) if r["removed"] else []
            st.write(f"Removed: {removed} | Success: {r['succeeds']} | Rule: {r['rule']}")

        st.write("**Best attenuation:**", payload["best"])

        # Plot decision tree
        st.pyplot(draw_attenuation_tree_for_goal(head, goal_rule["body"], goal_rule["body"], payload["results"]))
