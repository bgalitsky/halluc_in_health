import clingo
from collections import defaultdict

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


def explain_extensions(arguments, attacks, extensions):
    """Produce explanations for inclusion/exclusion of arguments."""
    # Build attack graph
    attackers = defaultdict(list)
    for x, y in attacks:
        attackers[y].append(x)

    explanations = []
    for ext in extensions:
        inc, exc = [], []
        for arg in arguments:
            if arg in ext:
                reason = []
                if attackers[arg]:
                    reason.append(f"survives despite being attacked by {attackers[arg]}")
                else:
                    reason.append("no attackers")
                inc.append((arg, ", ".join(reason)))
            else:
                if attackers[arg]:
                    reason = f"excluded because attacked by {attackers[arg]}"
                else:
                    reason = "excluded (not selected)"
                exc.append((arg, reason))
        explanations.append((ext, inc, exc))
    return explanations

if __name__ == "__main__":
    args = [
        "gout", "ra",
        "uric_acid", "tophi", "colchicine_response", "acute_onset",
        "symmetry", "rf_positive", "anti_ccp", "fever", "chronic_progression"
    ]

    atts = [
        # Mutual exclusivity
        ("gout", "ra"),
        ("ra", "gout"),

        # Gout evidence attacks RA
        ("uric_acid", "ra"),
        ("tophi", "ra"),
        ("colchicine_response", "ra"),
        ("acute_onset", "ra"),

        # RA evidence attacks Gout
        ("symmetry", "gout"),
        ("rf_positive", "gout"),
        ("anti_ccp", "gout"),
        ("chronic_progression", "gout"),
        ("fever", "gout"),
    ]

    for sem in ["stable", "preferred", "grounded"]:
        exts = compute_extensions(args, atts, semantics=sem)
        print(f"\n{sem.capitalize()} extensions: {exts}")

        if exts:
            explanations = explain_extensions(args, atts, exts)
            for ext, inc, exc in explanations:
                print(f"\nExtension {ext}:")
                print("  Included arguments:")
                for arg, reason in inc:
                    print(f"    - {arg}: {reason}")
                print("  Excluded arguments:")
                for arg, reason in exc:
                    print(f"    - {arg}: {reason}")
        else:
            print("No extension found.")
