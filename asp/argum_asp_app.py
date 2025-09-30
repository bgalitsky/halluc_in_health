import streamlit as st
import clingo
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------- ASP encodings ----------
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


# ---------- Solver ----------
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


# ---------- Explanations ----------
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


# ---------- Visualization ----------
def draw_graph(arguments, attacks):
    G = nx.DiGraph()
    G.add_nodes_from(arguments)
    G.add_edges_from(attacks)

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1500, arrowsize=20)
    nx.draw_networkx_edges(G, pos, edge_color="red", arrows=True)
    st.pyplot(plt.gcf())


# ---------- Streamlit UI ----------
st.title("🧠 Argumentation Framework Solver")

st.markdown("This tool lets you model **arguments** and their **attack relations**, "
            "then compute extensions under different **argumentation semantics**. "
            "Below you can expand each semantics to read its explanation:")

with st.expander("📘 Stable Semantics"):
    st.markdown("""
    **Stable semantics** selects sets of arguments that are:
    1. **Conflict-free** (no argument in the set attacks another in the set), and
    2. **Stable** in the sense that every argument *outside* the set is attacked by some argument *inside* the set.

    - **Intuition**: Stable extensions represent "self-defending" positions.  
    - **Pros**: Very decisive when they exist.  
    - **Cons**: They may not exist at all (e.g. in odd attack cycles).
    """)

with st.expander("📗 Preferred Semantics"):
    st.markdown("""
    **Preferred semantics** selects **maximal (by set inclusion) complete extensions**:
    - A set is **complete** if it is conflict-free and contains all arguments it defends.
    - A preferred extension is then a complete extension that cannot be extended further.

    - **Intuition**: Represents strong but possibly multiple "maximal" positions.  
    - **Pros**: Always exists. Captures plausible, strong stances.  
    - **Cons**: May return multiple, equally large extensions.
    """)

with st.expander("📙 Grounded Semantics"):
    st.markdown("""
    **Grounded semantics** selects the **minimal complete extension**:
    - It is the most skeptical approach, accepting only arguments that are **universally defendable**.

    - **Intuition**: Cautious baseline; what can be accepted without doubt.  
    - **Pros**: Always exists, unique.  
    - **Cons**: Can be very small (sometimes empty), missing stronger stances.
    """)


st.subheader("⚙️ Build Your Argumentation Framework")

args_input = st.text_area("Arguments (comma-separated)",
                          "gout, ra, uric_acid, tophi, colchicine_response, acute_onset, symmetry, rf_positive, anti_ccp, fever, chronic_progression")

atts_input = st.text_area("Attacks (one per line, format: attacker -> target)",
                          "gout -> ra\nra -> gout\nuric_acid -> ra\ntophi -> ra\ncolchicine_response -> ra\nacute_onset -> ra\nsymmetry -> gout\nrf_positive -> gout\nanti_ccp -> gout\nchronic_progression -> gout\nfever -> gout")

semantics = st.selectbox("Choose semantics", ["stable", "preferred", "grounded"])

if st.button("Compute Extensions"):
    arguments = [a.strip() for a in args_input.split(",") if a.strip()]
    attacks = []
    for line in atts_input.splitlines():
        if "->" in line:
            x, y = [p.strip() for p in line.split("->")]
            attacks.append((x, y))

    exts = compute_extensions(arguments, attacks, semantics=semantics)
    st.subheader(f"{semantics.capitalize()} Extensions")
    st.write(exts if exts else "No extensions found.")

    if exts:
        explanations = explain_extensions(arguments, attacks, exts)
        for ext, inc, exc in explanations:
            st.markdown(f"### Extension {ext}")
            st.write("**Included arguments:**")
            for arg, reason in inc:
                st.write(f"- {arg}: {reason}")
            st.write("**Excluded arguments:**")
            for arg, reason in exc:
                st.write(f"- {arg}: {reason}")

    st.subheader("Attack Graph")
    draw_graph(arguments, attacks)
