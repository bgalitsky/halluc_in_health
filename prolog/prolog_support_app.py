import streamlit as st
from openai import OpenAI
import pandas as pd
import subprocess
import tempfile
import time
from rule_attenuation_manager import AttenuatedReasoner, format_reasoning_output

from pyswip import Prolog

from pipeline import text_to_prolog_facts, question_to_prolog_query,  run_prolog_query, add_to_prolog_knowledge_base, analyze_ontology

# Initialize GPT client
client = OpenAI()
# --- Load CSV ---
#@st.cache_data
def load_data():
    ts = time.time()
    return pd.read_csv("data/autoimmune_diseases_with_complaints.csv",  encoding='cp1252')

df = load_data()

#---------------- Ontology ----------------
ontology_init = """inflammation(joints(A)) :- joints(A), member(A, [one,few,both,multiple,toe,knee,ankle])
inflammation(pain(S)) :- pain(S), member(S, [painfull,severe,throbbing,crushing,excruciating])
inflammation(property(C)) :- property(C), member(C, [red,warm,tender,swollen,fever])
inflammation(last(L)) :- last(L), member(L, [few_days,return,additional_longer])
disease(gout) :- inflammation(joints(A)), inflammation(pain(S)), inflammation(property(C)), inflammation(last(L))
"""

patient_facts_init = ["joints(toe)", "pain(severe)", "property(red)", "last(few_days)"]

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
# --- Lookup row matching selected complaint ---
matched_rows = df[df['sample_patient_complaint'] == user_query]

if not matched_rows.empty:
    row = matched_rows.iloc[0]               # ✅ Safe to access
    existing_ontology = row['disease_symptoms']  # ✅ Get value from Series
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
                ontology_text = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "give symptom description for the disease based on patient complaint: "+ user_query}]
                ).choices[0].message.content
            else:
                ontology_text = existing_ontology

            # 2. Ontology (textual)

            #client.chat.completions.create(
            #    model="gpt-4o",
            #    messages=[
            #        {"role": "system", "content": "Enumerate facts needed to answer user questions"},
            #        {"role": "user", "content": user_query}
            #    ]
            #).choices[0].message.content

            # 3. Ontology in Prolog
            ontology_prolog = ontology_init #text_to_prolog_facts(ontology_text)
            add_to_prolog_knowledge_base(ontology_prolog)

            # 4. Prolog query
            #list_of_predicates = extract_prolog_predicates(ontology_prolog)
            list_of_predicates, goal_predicate = analyze_ontology(ontology_prolog)
            query_prolog = question_to_prolog_query(user_query, ontology_prolog, list_of_predicates)

            # 5. Run Prolog query
            results = run_prolog_query(query_prolog, goal_predicate)
            reasoner = AttenuatedReasoner(ontology_prolog)
            atten_result = reasoner.run_w_attenuation(user_query, patient_facts_init)
            eliminated = []
            # Save in session
            st.session_state.gpt_response = ontology_text
            st.session_state.ontology_text = ontology_text
            st.session_state.ontology_prolog = ontology_prolog
            st.session_state.query_prolog = query_prolog
            results_str = [str(r) for r in results]
            eliminated_str = [str(r) for r in eliminated]

            st.session_state.result = (
                    "✅ Results:\n" + "\n".join(results_str) +
                    "\n"+format_reasoning_output(atten_result)
            )


# --- Output areas ---
st.subheader("1. GPT-5 Response")
st.text_area("GPT-5 Answer:", st.session_state.gpt_response, height=150, disabled=True)

st.subheader("2. Ontology (Textual)")
edited_ontology_text = st.text_area("Ontology (editable):", st.session_state.ontology_text, height=150)

st.subheader("3. Ontology in Prolog")
st.text_area("Prolog Ontology (editable)", st.session_state.ontology_prolog, height=150)

if st.button("Rerun with Edited Ontology"):
    prolog = Prolog()
    # regenerate Prolog ontology from edited text
    #ontology_prolog = text_to_prolog_facts(edited_ontology_text)
    ontology_prolog = st.session_state.ontology_prolog
    add_to_prolog_knowledge_base(ontology_prolog)
    print("ONTOLOGY PROLOG:" + ontology_prolog)
    list_of_predicates, goal_predicate = analyze_ontology(ontology_prolog)
    results = run_prolog_query(st.session_state.query_prolog, goal_predicate)

    eliminated = []
    st.session_state.ontology_text = edited_ontology_text
    st.session_state.ontology_prolog = ontology_prolog
    results_str = [str(r) for r in results]
    eliminated_str = [str(r) for r in eliminated]

    st.session_state.result = (
            "✅ Results:\n" + "\n".join(results_str)
            #"\n\n❌ Eliminated clauses:\n" + "\n".join(eliminated_str)
    )


#st.subheader("3. Ontology in Prolog")
#st.text_area("Prolog Ontology:", st.session_state.ontology_prolog, height=150, disabled=True)

st.subheader("4. Query in Prolog")
edited_query_prolog = st.text_area("Prolog Query (editable):", st.session_state.query_prolog, height=100)

if st.button("Rerun with Edited Query"):
    list_of_predicates, goal_predicate = analyze_ontology(st.session_state.ontology_prolog)
    print("About to run query: "+edited_query_prolog + " | goal ="+ goal_predicate)
    results = run_prolog_query(edited_query_prolog, goal_predicate)
    eliminated = []
    st.session_state.query_prolog = edited_query_prolog
    results_str = [str(r) for r in results]
    eliminated_str = [str(r) for r in eliminated]

    st.session_state.result = (
            "✅ Results:\n" + "\n".join(results_str)
            #"\n\n❌ Eliminated clauses:\n" + "\n".join(eliminated_str)
    )

st.subheader("5. Prolog Query Result")
st.text_area("Result:", st.session_state.result, height=150, disabled=True)


result_for_success = """{'facts': ['joints(toe)', 'pain(severe)', 'property(red)', 'last(few_days)'], 'goal': 'disease', 
'original_check': False, 'trace': {'joints': [{'A': 'toe'}], 'pain': [{'S': 'severe'}], 
'properties': [{'C': 'red'}], 'last': [{'L': 'few_days'}], 'disease': []}, 'results': [{'removed': (), 
'rule': 'disease :- inflammation(joints(A)), inflammation(pain(S)), inflammation(property(C)), inflammation(last(L))', 'succeeds': True, 'kept_count': 4}, 
{'removed': ('inflammation(pain(S))',), 'rule': 'disease :- inflammation(joints(A)), inflammation(property(C)), inflammation(last(L))', 'succeeds': True, 'kept_count': 3}, 
{'removed': ('inflammation(property(C))',), 'rule': 'disease :- inflammation(joints(A)), inflammation(pain(S)), inflammation(last(L))', 'succeeds': True, 'kept_count': 3}, 
{'removed': ('inflammation(property(C))',), 'rule': 'disease :- inflammation(joints(A)), inflammation(pain(S)), inflammation(last(L))', 'succeeds': True, 'kept_count': 3}, 
{'removed': ('inflammation(pain(S))', 'inflammation(property(C))'), 'rule': 'disease :- inflammation(joints(A)), inflammation(last(L))', 'succeeds': True, 'kept_count': 2},
 {'removed': ('inflammation(pain(S))', 'inflammation(property(C))'), 'rule': 'disease :- inflammation(joints(A)), inflammation(last(L))', 'succeeds': True, 'kept_count': 2},
  {'removed': ('inflammation(property(C))', 'inflammation(property(C))'), 'rule': 'disease :- inflammation(joints(A)), inflammation(pain(S)), inflammation(last(L))', 
  'succeeds': True, 'kept_count': 3}, 
  {'removed': ('inflammation(pain(S))', 'inflammation(property(C))', 'inflammation(property(C))'), 'rule': 'disease :- inflammation(joints(A)), inflammation(last(L))', 'succeeds': True, 'kept_count': 2}], 
  'best': {'removed': (), 'rule': 'disease :- inflammation(joints(A)), inflammation(pain(S)), inflammation(property(C)), inflammation(last(L))', 'succeeds': True, 'kept_count': 4}}
"""