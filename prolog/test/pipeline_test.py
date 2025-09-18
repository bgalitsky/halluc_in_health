from prolog.pipeline import analyze_ontology

ontology = """ 
pain(joint(A)) :- joint(A), member(A, [symmetrical, wrists, hands, feet]).
swelling(joint(A)) :- joint(A), member(A, [symmetrical, wrists, hands, feet]).
stiffness(morning(D)) :- morning(D), member(D, [gt_30_min]).
symptom(fatigue(S)) :- fatigue(S), member(S, [present]).
nodule(type(N)) :- type(N), member(N, [rheumatoid]).
inflammation(system(P)) :- system(P), member(P, [systemic]).
involvement(organ(O)) :- organ(O), member(O, [lung, eye]).
disease(rheumatoid_arthritis) :- pain(joint(A)), swelling(joint(A)), stiffness(morning(D)), symptom(fatigue(S)), nodule(type(N)), inflammation(system(P)).
disease(rheumatoid_arthritis) :- pain(joint(A)), swelling(joint(A)), stiffness(morning(D)), symptom(fatigue(S)), nodule(type(N)), inflammation(system(P)), involvement(organ(O)).
"""

pair = analyze_ontology(ontology)
print(pair)
