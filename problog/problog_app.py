import json
import streamlit as st
from problog.program import PrologString

from facts_clauses_to_problog import get_discourse_tree, discourse_to_problog, evaluate_with_timeout

st.title("Probabilistic Logic Program Builder")

default_facts = (
    "For the past few days in my knee and ankle pain was throbbing. "
    "When I woke up at night due to severe pain, I took a painkiller. "
    "When I carefully looked at my red joint, it seemed swollen. "
    "Once I discovered that I had a fever, I started thinking how to cure it."
)

default_ontology = (
    "Gout as a disease characterized by inflammation that must simultaneously involve four dimensions: "
    "(1) the joints affected, which may include one, a few, both, or multiple joints such as the toe, knee, or ankle; "
    "(2) the nature of pain, which can be painful, severe, throbbing, crushing, or excruciating; "
    "(3) observable inflammatory properties, such as redness, warmth, tenderness, swelling, or fever; and "
    "(4) the temporal pattern, which may last a few days, recur, or become prolonged. "
    "When inflammation is confirmed across these categories of joints, pain, properties, and duration, "
    "the condition is classified as gout."
)

ontology_text = st.text_area("Ontology / Clauses", value=default_ontology, height=200)
facts_text = st.text_area("Facts / Narrative", value=default_facts, height=200)

if st.button("Build Probabilistic Logic Program"):
    st.info("Calling discourse parser for both texts…")
    disc_clauses = get_discourse_tree(ontology_text)
    disc_facts = get_discourse_tree(facts_text)

    st.info("Building merged ProbLog program via GPT-5")
    merged_prog, extracted_query = discourse_to_problog(disc_clauses, disc_facts)

    st.subheader("Merged ProbLog Program")
    st.code(merged_prog, language="prolog")

    st.write("**Extracted query:**", extracted_query)

    try:
        st.info("Evaluating program…")
        res = evaluate_with_timeout(PrologString(merged_prog))
        st.subheader("Evaluation Results")
        for atom, prob in res.items():
            st.write(f"**{atom}**: {prob:.4f}")
    except Exception as e:
        st.error(f"Evaluation error: {e}")
