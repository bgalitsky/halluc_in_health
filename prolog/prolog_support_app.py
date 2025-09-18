import streamlit as st
from openai import OpenAI
import pandas as pd
import subprocess
import tempfile
import os

from pipeline import text_to_prolog_facts, question_to_prolog_query,  run_prolog_query, add_to_prolog_knowledge_base, analyze_ontology

# Initialize GPT client
client = OpenAI()
# --- Load CSV ---
@st.cache_data
def load_data():
    return pd.read_csv("data/autoimmune_diseases_with_complaints.csv")

df = load_data()


st.title("Ontology-Driven QA with GPT-5 + Prolog")

# Input

# Predefined symptom options (customize as needed)
#symptom_options = [
#"I have inflammation in my toe with sharp sudden onset of pain and elevated uric acid",
#"Joint Pain, symmetrical in wrists. Palpable swelling around joints. ",
#"What is my diagnosis if my inflamed joint is toe, my pain is severe, i have a fever and the disease is recurrent"
#]

# --- Create selectbox from sample_patient_complaint column ---
symptom_options = df['sample_patient_complaint'].tolist()

# Let user select from list
selected = st.selectbox("Choose a patient complaint", symptom_options, index = 0)

# Let user type custom — if they do, it overrides selection
custom = st.text_area("Or describe in your own words:", placeholder="e.g., I have wrist pain every morning...")

# Final description: use custom if provided, else selected
user_query = custom.strip() if custom.strip() else (selected if selected != "--- Type your own below ---" else None)

# --- Lookup row matching selected complaint ---
rows = df[df['sample_patient_complaint'] == user_query].iloc[0]
if not rows.empty:
    existing_ontology = rows.iloc[0]['disease_symptoms']
else:
    existing_ontology = None

#user_query =
#(
#    st.text_area(
#    "Enter your question:",
#    value="What is my diagnosis if my inflamed joint is toe, my pain is severe, i have a fever and the disease is recurrent"
#          #"I have inflammation in my toe with sharp sudden onset of pain and elvated uric acid"
#))

# Store intermediate states
if "ontology_text" not in st.session_state:
    st.session_state.ontology_text = ""
if "ontology_prolog" not in st.session_state:
    st.session_state.ontology_prolog = ""
if "query_prolog" not in st.session_state:
    st.session_state.query_prolog = ""
if "gpt_response" not in st.session_state:
    st.session_state.gpt_response = ""
if "result" not in st.session_state:
    st.session_state.result = ""


if st.button("Run GPT & Prolog pipeline"):
    if user_query.strip():
        with (st.spinner("Thinking...")):

            # 1. GPT natural response
            if existing_ontology is None:
                gpt_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "give symptom description for the disease based on patient complaint: "+ user_query}]
                ).choices[0].message.content

            # 2. Ontology (textual)
            ontology_text = """
            In most cases, only one or a few joints are affected. The big toe, knee, or ankle joints are most often affected.
            Sometimes many joints become swollen and painful.
            The pain starts suddenly, often during the night. Pain is often severe, described as throbbing, crushing, or excruciating.
            The joint appears warm and red. It is most often very tender and swollen (it hurts to put a sheet or blanket over it).
            There may be a fever.
            The attack may go away in a few days, but may return from time to time. Additional attacks often last longer.
            """

            #client.chat.completions.create(
            #    model="gpt-4o",
            #    messages=[
            #        {"role": "system", "content": "Enumerate facts needed to answer user questions"},
            #        {"role": "user", "content": user_query}
            #    ]
            #).choices[0].message.content

            # 3. Ontology in Prolog
            ontology_prolog = text_to_prolog_facts(ontology_text)
            add_to_prolog_knowledge_base(ontology_prolog)

            # 4. Prolog query
            #list_of_predicates = extract_prolog_predicates(ontology_prolog)
            list_of_predicates, goal_predicate = analyze_ontology(ontology_prolog)
            query_prolog = question_to_prolog_query(user_query, ontology_prolog, list_of_predicates)

            # 5. Run Prolog query
            results = run_prolog_query(query_prolog, goal_predicate)
            eliminated = []
            # Save in session
            st.session_state.gpt_response = gpt_response
            st.session_state.ontology_text = ontology_text
            st.session_state.ontology_prolog = ontology_prolog
            st.session_state.query_prolog = query_prolog
            results_str = [str(r) for r in results]
            eliminated_str = [str(r) for r in eliminated]

            st.session_state.result = (
                    "✅ Results:\n" + "\n".join(results_str) +
                    "\n\n❌ Eliminated clauses:\n" + "\n".join(eliminated_str)
            )



# --- Output areas ---
st.subheader("1. GPT-5 Response")
st.text_area("GPT-5 Answer:", st.session_state.gpt_response, height=150, disabled=True)

st.subheader("2. Ontology (Textual)")
edited_ontology_text = st.text_area("Ontology (editable):", st.session_state.ontology_text, height=150)

if st.button("Rerun with Edited Ontology"):
    # regenerate Prolog ontology from edited text
    ontology_prolog = text_to_prolog_facts(edited_ontology_text)
    add_to_prolog_knowledge_base(ontology_prolog)
    list_of_predicates, goal_predicate = analyze_ontology(ontology_prolog)
    results = run_prolog_query(st.session_state.query_prolog, goal_predicate)
    eliminated = []
    st.session_state.ontology_text = edited_ontology_text
    st.session_state.ontology_prolog = ontology_prolog
    results_str = [str(r) for r in results]
    eliminated_str = [str(r) for r in eliminated]

    st.session_state.result = (
            "✅ Results:\n" + "\n".join(results_str) +
            "\n\n❌ Eliminated clauses:\n" + "\n".join(eliminated_str)
    )


st.subheader("3. Ontology in Prolog")
st.text_area("Prolog Ontology:", st.session_state.ontology_prolog, height=150, disabled=True)

st.subheader("4. Query in Prolog")
edited_query_prolog = st.text_area("Prolog Query (editable):", st.session_state.query_prolog, height=100)

if st.button("Rerun with Edited Query"):
    list_of_predicates, goal_predicate = analyze_ontology(edited_ontology_text)
    results = run_prolog_query(edited_query_prolog, goal_predicate)
    eliminated = []
    st.session_state.query_prolog = edited_query_prolog
    results_str = [str(r) for r in results]
    eliminated_str = [str(r) for r in eliminated]

    st.session_state.result = (
            "✅ Results:\n" + "\n".join(results_str) +
            "\n\n❌ Eliminated clauses:\n" + "\n".join(eliminated_str)
    )

st.subheader("5. Prolog Query Result")
st.text_area("Result:", st.session_state.result, height=150, disabled=True)
