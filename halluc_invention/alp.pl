% =========================
% Hall2Invent: ALP Template
% =========================

% --- Observations / Goals ---
obs(no_external_control).
goal(passive_temp_regulation).  % replace per domain

% --- Abducibles (hypotheses/design choices) ---
% Declare abducibles by allowing them to be assumed.
% (Implementation depends on your ALP library; here we mark them with abducible/1.)
abducible(graded_catalyst).
abducible(high_k_inserts).
abducible(staged_feed).
abducible(active_cooling).
abducible(oscillatory_feedback).  % hallucinated candidate

% --- Background theory T (domain rules) ---
achieves(passive_temp_regulation) :-
    smooths(axial_heat_release),
    sufficient(heat_dissipation),
    no(active_control).

no(active_control) :- obs(no_external_control).

smooths(axial_heat_release) :- graded_catalyst.
smooths(axial_heat_release) :- staged_feed.
smooths(axial_heat_release) :- oscillatory_feedback.  % hallucination enters hypothesis space

sufficient(heat_dissipation) :- high_k_inserts.

% --- Safety / feasibility abstractions ---
runaway_risk :- not(smooths(axial_heat_release)), not(sufficient(heat_dissipation)).

% --- Integrity constraints IC (reject inadmissible hypotheses) ---
:- active_cooling, obs(no_external_control).  % violates passive requirement

unsupported(oscillatory_feedback).            % mark hallucinated mechanism unsupported (hard form)
:- oscillatory_feedback, unsupported(oscillatory_feedback).

:- runaway_risk.

% --- Query ---
% Find H ⊆ Abducibles such that T ∪ H entails achieves(Goal) and satisfies IC.
% In pure Prolog you’ll need an ALP engine (e.g., Abductive LP meta-interpreter) or encode search explicitly.
%
% Intended query:
% ?- achieves(passive_temp_regulation).

cost(graded_catalyst, 2).
cost(high_k_inserts, 3).
cost(staged_feed, 4).
cost(active_cooling, 10).
cost(oscillatory_feedback, 100).  % hallucination penalty
