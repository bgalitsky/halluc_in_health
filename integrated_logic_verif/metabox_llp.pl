lab(0.9, lab)   : infection(p1).
lab(0.8, vital) : fever(p1).

lab(0.85, guideline) : sepsis(p) :-
    infection(p), fever(p).

compatible(lab, guideline).
compatible(vital, guideline).

combine(conf(A,_), conf(B,_), conf(C,derived)) :-
    C is min(A,B).

requires(sepsis, 0.8).

accept(Head, conf(C,_)) :-
    requires(Head, T),
    C >= T.

lab(0.90, {monitor}) : hypotension(p1).
lab(0.85, {lab})     : lactate_high(p1).
lab(0.95, {monitor}) : tachycardia(p1).

lab(0.90, {guideline_shock}) : shock(P) :-
    hypotension(P), lactate_high(P).

lab(0.85, {guideline_sirs}) : sirs_like(P) :-
    tachycardia(P), infection(P).     % infection may be abduced

lab(0.90, {protocol_sepsis}) : sepsis_risk(P) :-
    shock(P), sirs_like(P).

% (integrity constraint)
:- infection(P), confirmed_no_infection(P).

lab(0.60, {assumption_infection}) : infection(p1).