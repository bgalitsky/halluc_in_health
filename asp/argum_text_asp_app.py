from configparser import RawConfigParser
from functools import lru_cache
from joblib import Memory
import streamlit as st
import clingo
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from openai import OpenAI
import os

# Get directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build full path to config.ini relative to this script
config_path = os.path.join(script_dir, 'config.ini')


# Load configuration
config = RawConfigParser()
config.read(config_path) #'config.ini')

# API keys

os.environ["OPENAI_API_KEY"] = config.get('OpenAI', 'api_key')
# Cache directory persists between runs
memory = Memory("./cache", verbose=1)
# ---------------- GPT-5 Extraction ----------------
#@lru_cache(maxsize=128)
#@memory.cache
from openai import OpenAI

def extract_af_from_text(
    disease_text: str,
    symptom_text: str,
    *,
    model: str = "gpt-5",
    max_output_tokens: int = 800,
    reasoning_effort: str = "medium"   # "minimal" | "low" | "medium" | "high"
) -> str:
    """
    Returns a strictly formatted block like:

    Arguments:
    arg1, arg2, arg3, ...

    Attacks:
    argX -> argY
    argA -> argB
    ...
    """
    client = OpenAI()  # uses OPENAI_API_KEY from env

    instructions = (
        "You are an expert in computational argumentation and medical reasoning. "
        "Extract arguments (diseases + key symptoms/features) and enumerate attack "
        "relations between arguments (e.g., a feature that supports Disease A "
        "attacks Disease B). Output must match the exact format shown."
    )

    # Keep the user payload minimal; put durable behavior in `instructions`
    user_input = (
        f"Candidate diseases: {disease_text}\n"
        f"Patient symptoms: {symptom_text}\n\n"
        "Format the output strictly as:\n\n"
        "Arguments:\n"
        "arg1, arg2, arg3, ...\n\n"
        "Attacks:\n"
        "argX -> argY\n"
        "argA -> argB\n"
        "..."
    )

    print("instructions = " + instructions)
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": reasoning_effort},   # optional, GPT-5 supports this
        # temperature=0.2,                       # optional: make extra deterministic
    )

    # Responses API gives you a convenience string:
    text = (resp.output_text or "").strip()
    if text:
        return text

        # Fallback: collect text from structured output steps
    parts = []
    for step in resp.output:
        if step.type == "message":
            for content in step.content:
                if content.type == "output_text":
                    parts.append(content.text)
    print("\n".join(parts).strip())
    return "\n".join(parts).strip()




# ---------------- Parsing from text ----------------
def parse_af_from_text(content):
    args, atts = [], []
    in_atts = False
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Arguments:"):
            args = [a.strip() for a in line[len("Arguments:"):].split(",") if a.strip()]
        elif line.startswith("Attacks:"):
            in_atts = True
        elif in_atts and "->" in line:
            x, y = [p.strip() for p in line.split("->")]
            atts.append((x, y))
    return args, atts


# ---------------- ASP Encodings ----------------
def asp_encoding(semantics: str) -> str:
    base = """
    { in(X) : arg(X) }.
    :- in(X), in(Y), att(X,Y).
    """

    if semantics == "stable":
        return base + """
        attacked(X) :- in(Y), att(Y,X).
        :- arg(X), not in(X), not attacked(X).
        #show in/1.
        """
    elif semantics == "grounded":
        return base + """
        out(X) :- in(Y), att(Y,X).
        undec(X) :- arg(X), not in(X), not out(X).
        #show in/1.
        """
    elif semantics == "preferred":
        return base + """
        defended(X) :- arg(X), not att(Y,X) : in(Y).
        :- in(X), not defended(X).
        #maximize { 1,X : in(X) }.
        #show in/1.
        """
    else:
        raise ValueError(f"Unknown semantics: {semantics}")


# ---------------- Solver ----------------
def compute_extensions(arguments, attacks, semantics="stable"):
    asp_program = []
    for a in arguments:
        asp_program.append(f"arg({a}).")
    for x, y in attacks:
        asp_program.append(f"att({x},{y}).")
    asp_program.append(asp_encoding(semantics))

    program_text = "\n".join(asp_program)
    ctl = clingo.Control()
    ctl.add("base", [], program_text)
    ctl.ground([("base", [])])

    extensions = []

    def on_model(model):
        ins = [str(sym.arguments[0]) for sym in model.symbols(shown=True)]
        extensions.append(ins)

    ctl.solve(on_model=on_model)
    return extensions


# ---------------- Explanations ----------------
def explain_extensions(arguments, attacks, extensions):
    attackers = defaultdict(list)
    for x, y in attacks:
        attackers[y].append(x)

    explanations = []
    for ext in extensions:
        inc, exc = [], []
        for arg in arguments:
            if arg in ext:
                reason = "no attackers" if not attackers[arg] else f"survives despite being attacked by {attackers[arg]}"
                inc.append((arg, reason))
            else:
                if attackers[arg]:
                    reason = f"excluded because attacked by {attackers[arg]}"
                else:
                    reason = "excluded (not selected)"
                exc.append((arg, reason))
        explanations.append((ext, inc, exc))
    return explanations


# ---------------- Visualization ----------------
def draw_graph(arguments, attacks, extension=None, undecided=None):
    G = nx.DiGraph()
    G.add_nodes_from(arguments)
    G.add_edges_from(attacks)

    color_map = []
    for node in G.nodes():
        if extension and node in extension:
            color_map.append("lightgreen")
        elif undecided and node in undecided:
            color_map.append("lightgray")
        else:
            color_map.append("lightcoral")

    plt.figure(figsize=(7, 7))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=color_map,
            node_size=1500, font_size=10, arrowsize=20,
            edge_color="red")
    st.pyplot(plt.gcf())


# ---------------- Streamlit UI ----------------
st.title("🧠 Medical Argumentation Framework Solver with Editable Logic Program")

disease_text = st.text_area("Describe candidate diseases",
                            "Gout, Rheumatoid Arthritis, Osteoarthritis")
symptom_text = st.text_area("Describe patient symptoms",
                            "Sudden swelling of toe, elevated uric acid, "
                            "symmetry of hand joints, chronic stiffness, "
                            "positive rheumatoid factor, cartilage wear on X-ray")

semantics = st.selectbox("Choose semantics", ["stable", "preferred", "grounded"])

if st.button("Extract with GPT-5"):
    raw_output = extract_af_from_text(disease_text, symptom_text)
    st.session_state["af_text"] = raw_output  # store in session

# Editable AF text area
af_text = st.text_area("Edit Arguments and Attacks",
                       st.session_state.get("af_text", ""),
                       height=250)

if st.button("Run Solver"):
    arguments, attacks = parse_af_from_text(af_text)
    exts = compute_extensions(arguments, attacks, semantics=semantics)
    st.subheader(f"{semantics.capitalize()} Extensions")
    st.write(exts if exts else "No extensions found.")

    if exts:
        explanations = explain_extensions(arguments, attacks, exts)
        ext_options = [", ".join(ext) if ext else "∅ (empty)" for ext, _, _ in explanations]
        chosen_ext_label = st.selectbox("Select an extension to visualize", ext_options)
        chosen_ext_idx = ext_options.index(chosen_ext_label)
        chosen_ext, inc, exc = explanations[chosen_ext_idx]

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.markdown("### Explanation")
            st.write("**Included arguments:**")
            for arg, reason in inc:
                st.write(f"- {arg}: {reason}")
            st.write("**Excluded arguments:**")
            for arg, reason in exc:
                st.write(f"- {arg}: {reason}")

        with col2:
            st.markdown("### Attack Graph")
            draw_graph(arguments, attacks, extension=chosen_ext)
