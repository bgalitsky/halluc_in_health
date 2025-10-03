import streamlit as st
from rule_attenuation_manager import (
    AttenuatedReasoner,
    format_reasoning_output,
    ontology  # or you can just ask the user to paste ontology
)

st.title("Rule Attenuation Manager")

# Two text areas for input
ontology_text = st.text_area(
    "Ontology (Prolog clauses):",
    value=ontology,
    height=200
)

symptoms_text = st.text_area(
    "Complaint / Symptoms text:",
    value="For the past few days in my knee and ankle pain was throbbing. "
          "When I woke up at night due to severe pain, I took a painkiller. "
          "When I carefully looked at my red joint, it seemed swollen. "
          "Once I discovered that I had a fever, I started thinking how to cure it.",
    height=150
)

# Default patient facts (you can let user type these too)
patient_facts_str = st.text_area(
    "Patient facts (one per line, Prolog-style):",
    value="joints(toe)\npain(severe)\nproperty(red)\nlast(few_days)",
    height=100
)
patient_facts = [line.strip() for line in patient_facts_str.splitlines() if line.strip()]

if st.button("Run Attenuation"):
    try:
        reasoner = AttenuatedReasoner(ontology_text)
        result = reasoner.run_w_attenuation(symptoms_text, patient_facts)
        if result:
            st.subheader("Reasoning Summary")
            st.text(format_reasoning_output(result))
        else:
            st.warning("No result returned. Check your ontology and facts.")
    except Exception as e:
        st.error(f"Error running attenuation: {e}")
