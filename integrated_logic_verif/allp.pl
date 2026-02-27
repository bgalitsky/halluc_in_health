lab(0.9, lab) : infection(p1).
lab(0.8, vital) : fever(p1).

lab(0.85, guideline) : sepsis(p) :-
    infection(p), fever(p).

% Combination rule
combine(conf(A,_), conf(B,_), conf(C,derived)) :-
    C is min(A,B).

% Source compatibility
compatible(lab, guideline).
compatible(vital, guideline).

% Conflict resolution
prefer(guideline, nurse).

propagate(RuleLab, BodyLabs, HeadLab) :-
    compatible_all(RuleLab, BodyLabs),
    min_confidence([RuleLab|BodyLabs], HeadLab).

lab(0.9, lab)   : pneumonia(p1).
lab(0.7, nurse) : no_pneumonia(p1).

prefer(lab, nurse).
pneumonia(p1)  % accepted
no_pneumonia(p1) % rejected

lab(obs) : fever(p1).
lab(bel) : infection(p1).

lab(rec) : start_antibiotics(p) :-
    fever(p), infection(p).

lab(t1) : hypotension(p1).
lab(t3) : stable_bp(p1).