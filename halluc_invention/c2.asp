% =========================
% Hall2Invent: ASP Template
% =========================

% --- Choose abducibles (hypotheses) ---
{ graded_catalyst; high_k_inserts; staged_feed; active_cooling; oscillatory_feedback }.

% --- Observations / goals ---
obs(no_external_control).
goal(passive_temp_regulation).

% --- Background theory ---
smooths_axial_heat :- graded_catalyst.
smooths_axial_heat :- staged_feed.
smooths_axial_heat :- oscillatory_feedback.  % hallucinated candidate

sufficient_dissipation :- high_k_inserts.
no_active_control :- obs(no_external_control).

achieves_passive_reg :-
    smooths_axial_heat,
    sufficient_dissipation,
    no_active_control.

% --- Integrity constraints (hard) ---
:- active_cooling, obs(no_external_control).

unsupported(oscillatory_feedback).
:- oscillatory_feedback, unsupported(oscillatory_feedback).

% runaway if no smoothing AND no dissipation
runaway_risk :- not smooths_axial_heat, not sufficient_dissipation.
:- runaway_risk.

% --- Ensure goal achieved ---
:- not achieves_passive_reg.

% --- Costs (prefer feasible, low-cost hypotheses) ---
:~ graded_catalyst. [2@1]
:~ high_k_inserts.  [3@1]
:~ staged_feed.     [4@1]
:~ active_cooling.  [10@1]
:~ oscillatory_feedback. [100@1]

% --- Output ---
#show graded_catalyst/0.
#show high_k_inserts/0.
#show staged_feed/0.
#show active_cooling/0.
#show oscillatory_feedback/0.
