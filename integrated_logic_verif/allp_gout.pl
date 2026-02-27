% --- Abducible diagnoses ---
abducible(gout(P)).
abducible(ra(P)).

% --- Integrity constraint: treat them as mutually exclusive for this toy verifier ---
:- gout(P), ra(P).

lab(0.95, {text_extract}) : acute_onset(p1).
lab(0.90, {text_extract}) : monoarthritis(p1).
lab(0.85, {text_extract}) : first_mtp_involved(p1).   % podagra cue
lab(0.80, {text_extract}) : alcohol_trigger(p1).
lab(0.85, {lab_value})    : serum_urate_high(p1).

% Classic gout phenotype (acute, monoarticular, first MTP)
lab(0.90, {guideline_gout}) : explains_gout_pattern(P) :-
    gout(P),
    acute_onset(P),
    monoarthritis(P),
    first_mtp_involved(P).

% Hyperuricemia supports gout but is not definitive
lab(0.80, {guideline_urate}) : supports_gout(P) :-
    gout(P),
    serum_urate_high(P).

% If gout pattern holds, propose diagnosis candidate
lab(0.90, {dx_rule}) : dx(gout,P) :-
    explains_gout_pattern(P).

% Typical RA phenotype (symmetric, polyarticular, prolonged morning stiffness)
lab(0.90, {guideline_ra}) : explains_ra_pattern(P) :-
    ra(P),
    symmetric_polyarthritis(P),
    morning_stiffness_60min(P).

% Anti-CCP is strong support when present
lab(0.95, {guideline_ccp}) : supports_ra(P) :-
    ra(P),
    anti_ccp_pos(P).

lab(0.90, {dx_rule}) : dx(ra,P) :-
    explains_ra_pattern(P).

lab(0.85, {dx_rule_ccp}) : dx(ra,P) :-
    supports_ra(P).

% These are not “given facts”; they are what abduction *may* assume when needed.
% (Shown explicitly here for clarity.)
lab(0.60, {assumption}) : gout(p1).
lab(0.60, {assumption}) : ra(p1).

abducible(symmetric_polyarthritis(P)).
abducible(morning_stiffness_60min(P)).
abducible(anti_ccp_pos(P)).

% Typical weak labels for abduced symptoms/tests:
lab(0.45, {assumption}) : symmetric_polyarthritis(p1).
lab(0.45, {assumption}) : morning_stiffness_60min(p1).
lab(0.40, {assumption}) : anti_ccp_pos(p1).