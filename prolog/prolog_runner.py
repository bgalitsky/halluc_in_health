from pyswip import Prolog

# Initialize Prolog
prolog = Prolog()

# Add facts/rules directly in Python
prolog.assertz("father(tom, bob)")
prolog.assertz("father(tom, liz)")
prolog.assertz("mother(ann, bob)")

# Define a rule
prolog.assertz("parent(X, Y) :- father(X, Y)")
prolog.assertz("parent(X, Y) :- mother(X, Y)")

# Query
result = list(prolog.query("parent(tom, Child)"))
print(result)  # Output: [{'Child': 'bob'}, {'Child': 'liz'}]