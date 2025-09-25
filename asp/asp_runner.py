import clingo

def compute_stable_extensions(arguments, attacks):
    """
    Compute stable extensions of a Dung argumentation framework
    using the clingo Python API (no external binary).
    """

    # Build ASP program
    asp_program = []

    # Arguments
    for a in arguments:
        asp_program.append(f"arg({a}).")   # must be lowercase, no quotes

    # Attacks
    for x, y in attacks:
        asp_program.append(f"att({x},{y}).")

    # Stable semantics encoding
    asp_program += [
        "{ in(X) : arg(X) }.",                  # guess subset
        ":- in(X), in(Y), att(X,Y).",           # conflict-free
        "attacked(X) :- in(Y), att(Y,X).",
        ":- arg(X), not in(X), not attacked(X).",  # outsiders must be attacked
        "#show in/1."
    ]

    program_text = "\n".join(asp_program)
    print("ASP program:\n", program_text)   # DEBUG

    ctl = clingo.Control()
    ctl.add("base", [], program_text)
    ctl.ground([("base", [])])

    extensions = []

    def on_model(model):
        # collect in/1 atoms
        ins = [str(sym.arguments[0]) for sym in model.symbols(shown=True)]
        extensions.append(ins)

    result = ctl.solve(on_model=on_model)
    print("Solver result:", result)  # should be SAT
    return extensions


if __name__ == "__main__":
    args = ["a", "b"]
    atts = [("a", "b")]

    exts = compute_stable_extensions(args, atts)
    print("Stable extensions:", exts)
