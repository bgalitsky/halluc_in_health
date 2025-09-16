import streamlit as st
from openai import OpenAI
import subprocess
import tempfile
import os

from pipeline import text_to_prolog_facts, question_to_prolog_query, extract_prolog_predicates, run_prolog_query, \
    add_to_prolog_knowledge_base, run_prolog_query_relaxed

# Initialize GPT client
client = OpenAI()


st.title("Ontology-Driven QA with GPT-5 + Prolog")

# Input
user_query = st.text_area(
    "Enter your question:",
    value="What is my diagnosis if I have inflammation in my toe with sharp sudden onset of pain and elvated uric acid"
)

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
        with st.spinner("Thinking..."):

            # 1. GPT natural response
            gpt_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": user_query}]
            ).choices[0].message.content

            # 2. Ontology (textual)
            ontology_text = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Enumerate facts needed to answer user questions"},
                    {"role": "user", "content": user_query}
                ]
            ).choices[0].message.content

            # 3. Ontology in Prolog
            ontology_prolog = text_to_prolog_facts(ontology_text)
            add_to_prolog_knowledge_base(ontology_prolog)

            # 4. Prolog query
            list_of_predicates = extract_prolog_predicates(ontology_prolog)
            query_prolog = question_to_prolog_query(user_query, ontology_prolog, list_of_predicates)

            # 5. Run Prolog query
            results, eliminated = run_prolog_query_relaxed(query_prolog)

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
    result = run_prolog_query(st.session_state.query_prolog)
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
    results, eliminated = run_prolog_query_relaxed(edited_query_prolog)
    st.session_state.query_prolog = edited_query_prolog
    results_str = [str(r) for r in results]
    eliminated_str = [str(r) for r in eliminated]

    st.session_state.result = (
            "✅ Results:\n" + "\n".join(results_str) +
            "\n\n❌ Eliminated clauses:\n" + "\n".join(eliminated_str)
    )

st.subheader("5. Prolog Query Result")
st.text_area("Result:", st.session_state.result, height=150, disabled=True)
