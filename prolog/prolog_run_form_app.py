import streamlit as st
from pyswip import Prolog

from pipeline import add_to_prolog_knowledge_base

# Initialize Prolog
#  nohup streamlit run prolog_run_form_app.py --server.port 8501 --server.address 0.0.0.0 &

prolog = Prolog()

st.title("Diagnosis Prolog App")

st.markdown("This app uses a Prolog ontology and patient complaints to check for **disease(gout)**.")

# Editable ontology window
default_ontology = """% Ontology rules for gout

inflammation(joints(A)) :- joints(A), member(A, [one, few, both, multiple, toe, knee, ankle]).
inflammation(pain(S)) :- pain(S), member(S, [painfull, severe, throbbing, crushing, excruciating]).
inflammation(property(C)) :- property(C), member(C, [red, warm, tender, swollen, fever]).
inflammation(last(L)) :- last(L), member(L, [few_days, return, additional(longer)]).

% Disease definition
disease(gout) :- inflammation(joints(A)),  inflammation(pain(S)), inflammation(property(C)), inflammation(last(L)).
"""

ontology_input = st.text_area("Ontology (editable):", default_ontology, height=200)

# Patient complaint (facts)
facts_input = st.text_area("Enter patient complaint or car repair problem as Prolog facts:",
"""joints(toe).
pain(painfull).
property(red).
property(warm).
last(few_days).""")

if st.button("Run Diagnosis"):
    try:
        # Reload Prolog completely
        prolog = Prolog()
        # ---- Clean ontology ----
        # Ensure no previous predicates remain
        prolog.assertz("clean_start(true)")  # dummy to initialize
        prolog.retractall("inflammation(_)")
        prolog.retractall("disease(_)")

        add_to_prolog_knowledge_base(ontology_input)


        # Split input into facts
        facts = [f.strip().rstrip('.') for f in facts_input.split("\n") if f.strip()]

        # Assert facts temporarily
        for fact in facts:
            prolog.assertz(fact)

        # Run query
        results = list(prolog.query("disease(D)"))

        # Retract facts after query
        for fact in facts:
            prolog.retract(fact)

        # Output
        if results:
            st.success(f"Diagnosis matched: {results}")
        else:
            st.warning("No diagnosis matched from the ontology.")

    except Exception as e:
        st.error(f"Error: {e}")
