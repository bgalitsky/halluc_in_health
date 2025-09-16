from pyswip import Prolog
from openai import OpenAI
from configparser import RawConfigParser
import os
from functools import lru_cache
from joblib import Memory
import itertools

# Cache directory persists between runs
#memory = Memory("./cache", verbose=1)
memory = Memory("./llm_prolog_cache", verbose=1, compress=1)

# Load configuration
config = RawConfigParser()
config.read('config.ini')

# API keys

os.environ["OPENAI_API_KEY"] = config.get('OpenAI', 'api_key')


MODEL_NAME = "gpt-5"  # or "gpt-3.5-turbo"
client = OpenAI()

prolog = Prolog()  # Initialize Prolog engine

# -------------------------------
# Step 2: Use GPT to Convert Text → Prolog Facts/Rules
# -------------------------------
@lru_cache(maxsize=128)
@memory.cache
def text_to_prolog_facts(text):
    prompt = f"""
    Convert the following natural language text into logical Prolog facts and rules.
    Use lowercase predicate names and atoms. Use single noun (subject) with a concrete meanining or linguistic predicate for predicate name.
    Use variables (starting with uppercase) in rules.
    Only output valid Prolog code, one fact/rule per line.
    Example:
     % Ontology rules for gout
    
    inflammation(joints(A)) :- joints(A), member(A, [one, few, both, multiple, toe, knee, ankle]).
    inflammation(pain(S)) :- pain(S), member(S, [painfull, severe, throbbing, crushing, excruciating]).
    inflammation(property(C)) :- property(C), member(C, [red, warm, tender, swollen, fever]).
    inflammation(last(L)) :- last(L), member(L, [few_days, return, additional(longer)]).
    
    % Disease definition
    disease(gout) :- inflammation(joints(A)),  inflammation(pain(S)), inflammation(property(C)), inflammation(last(L)).

    Now process this text:
    "{text}"
    """

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        temperature=1
    )


    prolog_code = response.choices[0].message.content.strip()
    print("🔧 Generated Prolog facts/rules:")
    print(prolog_code)
    print("-" * 50)
    return prolog_code

# -------------------------------
# Step 3: Add Prolog Code to KB
# -------------------------------
def add_to_prolog_knowledge_base(prolog_code):
    lines = [line.strip() for line in prolog_code.split('\n') if line.strip() and not line.strip().startswith('%')]
    for line in lines:
        try:
            # Remove trailing period if present (assertz adds it)
            fact = line.rstrip('. ')
            prolog.assertz(fact)
        except Exception as e:
            print(f"⚠️ Failed to assert: {line} | Error: {e}")



# -------------------------------
# Step 5: Convert Question → Prolog Query Goal
# -------------------------------

def question_to_prolog_query(symptoms, ontology, list_of_predicates):
    prompt = f"""
    Represent a list of symptoms such as a patient complains in prolog form as a list of facts each ending with '.'
    Use only predicates from specified list {list_of_predicates}

    ontology that should be satisfied by these facts:
    {ontology}
    
    Symptoms: {symptoms}
    """

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        temperature=1
    )

    prolog_query = response.choices[0].message.content.strip()
    # Ensure it ends with a dot
    if not prolog_query.endswith('.'):
        prolog_query += '.'
    print("🔍 Prolog query goal:")
    print(prolog_query)
    print("-" * 50)
    return prolog_query.rstrip('.')

# -------------------------------
# Step 6: Run Query in Prolog
# -------------------------------
def run_prolog_query(query):
    try:
        # Split input into facts
        facts = [f.strip().rstrip('.') for f in query.split("\n") if f.strip()]
        # Assert facts temporarily
        for fact in facts:
            prolog.assertz(fact)

        results = list(prolog.query("disease(D)"))
        # todo: auto extract this goal from left-bottom corner of ontology

        # Retract facts after query
        for fact in facts:
            prolog.retract(fact)
        if results:
            print("✅ Answer (Prolog results):")
            for result in results:
                print(result)
            return results
        else:
            print("❌ No solutions found for the query.")
            return []
    except Exception as e:
        print(f"❌ Prolog execution error: {e}")
        return []


def split_prolog_goals(query: str):
    """
    Splits a Prolog query into top-level goals, ignoring commas inside parentheses.
    """
    goals = []
    current = ""
    depth = 0
    for c in query:
        if c == "(":
            depth += 1
            current += c
        elif c == ")":
            depth -= 1
            current += c
        elif c == "," and depth == 0:
            # top-level comma, split here
            if current.strip():
                goals.append(current.strip())
            current = ""
        else:
            current += c
    if current.strip():
        goals.append(current.strip())
    return goals

def run_prolog_query_relaxed(query: str):
    """
    Iterative relaxation removing any goals until query is satisfied.
    Returns:
        results: first Prolog solutions found
        eliminated: list of goals removed from the original query
    """
    goals = split_prolog_goals(query)
    n = len(goals)

    for size in range(n, 0, -1):
        for subset in itertools.combinations(goals, size):
            subquery = ", ".join(subset)
            try:
                results = list(prolog.query(subquery))
                if results:
                    eliminated = [g for g in goals if g not in subset]
                    return results, eliminated
            except Exception as e:
                print(f"❌ Prolog execution error: {e}")

    return [], goals  # no solution


def get_predicate_signatures():
    """
    Queries the current SWI-Prolog knowledge base (via pyswip)
    and returns a list of predicate signatures in the form 'name/arity',
    e.g., ['loves/2', 'man/1', 'has_dog/2'].

    Filters out built-in and system predicates.
    """

    # Query for all defined predicates
    query_term = "current_predicate(FullSpec), predicate_property(Pred, user)"

    signatures = set()  # Use set to avoid duplicates

    try:
        for sol in Prolog().query(query_term):
            full_spec = sol["FullSpec"]  # Comes in format: module:Name/Arity or Name/Arity

            # Handle module prefix if present (e.g., user:loves/2)
            if ':' in str(full_spec):
                name_arity = str(full_spec).split(':', 1)[1]
            else:
                name_arity = str(full_spec)

            # Parse name and arity: e.g., loves/2
            if '/' in name_arity:
                try:
                    name, arity = name_arity.split('/')
                    arity = int(arity)
                    # Optional: filter out low-use or internal names
                    if name and name[0].islower() and not name.startswith('_'):
                        signatures.add(f"{name}/{arity}")
                except ValueError:
                    continue
    except Exception as e:
        print(f"⚠️ Error querying Prolog predicates: {e}")

    return sorted(signatures)  # Return sorted list for consistency

import re
from typing import List, Set

import re
from typing import List, Set


def extract_prolog_predicates(text: str) -> List[str]:
    """
    Extracts ONLY the head predicates from Prolog clauses (facts/rules).
    Avoids false positives from nested terms like a(b(c)).

    Returns list of 'name/arity' strings.
    """
    predicates: Set[str] = set()

    # Remove comments
    text = re.sub(r'%.*?\n', '\n', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Split into clauses using '.' as delimiter
    clauses = re.split(r'\.\s*', text)

    # Pattern to match clause head: starts at beginning, before ':-' or end of clause
    head_pattern = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)')

    for clause in clauses:
        clause = clause.strip()
        if not clause or clause.startswith(':') or clause == ';':
            continue

        # Extract head: everything before ':-'
        head_part = clause.split(':-')[0].strip()

        # Match function pattern: name(args)
        match = head_pattern.match(head_part)
        if match:
            pred_name = match.group(1)
            args_str = match.group(2).strip()

            # Count arguments at top level (ignore commas in nested parentheses)
            if args_str == "":
                arity = 0
            else:
                arity = count_top_level_commas(args_str) + 1
            predicates.add(f"{pred_name}/{arity}")
        else:
            # Might be a zero-arity fact: e.g., sunny.
            if re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*$', head_part):
                predicates.add(f"{head_part.strip()}/0")

    return sorted(predicates)


def count_top_level_commas(s: str) -> int:
    """Count commas not inside any parentheses."""
    depth = 0
    count = 0
    for ch in s.strip():
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        elif ch == ',' and depth == 0:
            count += 1
    return count


def count_top_level_commas(s: str) -> int:
    """
    Count commas at top level (not inside nested parentheses).
    """
    depth = 0
    count = 0
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            count += 1
    return count


# Wrapper: Read from file
def get_predicates_from_file(filename: str) -> List[str]:
    """
    Reads a .pl file and returns list of predicate signatures.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return extract_prolog_predicates(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")
    except Exception as e:
        raise RuntimeError(f"Error reading/parsing file: {e}")

def read_prolog_file(filename='ontology.pl'):
    """
    Reads a Prolog file and returns its content as a string.

    Args:
        filename (str): Path to the .pl file. Default is 'ontology.pl'.

    Returns:
        str: The full content of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prolog file '{filename}' not found. Please check the file path.")
    except Exception as e:
        raise IOError(f"Error reading file '{filename}': {e}")
# -------------------------------
# MAIN PIPELINE
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Intelligent Reasoning Pipeline\n")

    # Step 1
    text = """
    Feature	Gout	Rheumatoid Arthritis (RA)
Cause	Uric acid crystals	Autoimmune
Onset	Sudden (hours)	Gradual (weeks/months)
Pain Peak	First 24 hours	Builds over time
Joint Pattern	Often monoarticular	Symmetrical, polyarticular
Common Joints	Big toe, ankle, knee	Hands, wrists, feet (small joints)
Morning Stiffness	Minimal or none	>1 hour
Systemic Symptoms	Rare (unless severe)	Common (fatigue, fever, malaise)
Tophi/Nodules	Tophi (chalky, under skin)	Rheumatoid nodules (firm, elbows)
Blood Tests	High uric acid (not always)	RF, anti-CCP positive
Imaging	Bone erosion with overhanging edge	Symmetric joint space narrowing, erosions
"""

    # Step 2 & 3
    #prolog_kb = text_to_prolog_facts(text)
    prolog_kb = read_prolog_file()
    add_to_prolog_knowledge_base(prolog_kb)

    # Step 4
    question = (#"do I have a gout if "
                #"I have Cause:Uric acid crystals,
        "Common Joints:ankle, Tophi chalky, Joint Pattern:often_monoarticular")
    # Step 5
    list_of_predicates = extract_prolog_predicates(prolog_kb)
    #get_predicate_signatures()
    #query = question_to_prolog_query(question, text, list_of_predicates)
    query = "disease(D), joint_pattern(D, monoarticular), common_joint(D, ankle), systemic_symptoms_frequency(D, rare_unless_severe), lesion_type(D, tophi), lesion_characteristic(D, chalky_under_skin), lesion_common_site(D, elbows)."


    # Step 6
    #run_prolog_query(query)
    results, eliminated = run_prolog_query_relaxed(query)
    print(results)
    print(eliminated)

    print("\n✅ Pipeline finished.")