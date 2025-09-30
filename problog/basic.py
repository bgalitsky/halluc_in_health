from problog.program import PrologString
from problog.core import ProbLog
from problog.sdd_formula import SDD

p = PrologString("""coin(c1). coin(c2).
0.4::heads(C); 0.6::tails(C) :- coin(C).
win :- heads(C).
query(win).
""")

print(ProbLog.convert(p, SDD).evaluate())