:- module(alp_dx, [
    explain_obs/2,        % explain_obs(+ObsList, -Delta)
    explain_obs_k/3,      % explain_obs_k(+ObsList, +K, -Delta)
    assume_obs/1,         % assume_obs(+ObsList)
    clear_obs/0,          % clear previously asserted obs/1
    add_ic/1,             % add_ic(+ForbiddenLiteralsList)
    clear_ics/0,          % clear integrity constraints
    abducible/1,          % declare abducible functor/arity
    load_demo/0
]).

:- dynamic abducible/1.
:- dynamic ic/1.
:- dynamic obs/1.
:- dynamic causes/2.

% ------------------------ Abductive core (minimal by #abduced) ------------------------

explain_obs(ObsList, Delta) :-
    between(0, inf, K),
    explain_obs_k(ObsList, K, Delta), !.

explain_obs_k(ObsList, K, Delta) :-
    goals_from_obs(ObsList, Goal),
    prove(Goal, [], D0, K),
    valid_ic(D0),
    sort(D0, Delta).

goals_from_obs([], true).
goals_from_obs([S|Ss], (obs(S), Gs)) :- goals_from_obs(Ss, Gs).

prove(true, D, D, _) :- !.
prove((A,B), D0, D, K) :- !,
    prove(A, D0, D1, K),
    prove(B, D1, D,  K).

% Observation must be covered by some abduced (or abducible) disease via causes/2
prove(obs(S), D0, D, K) :- !,
    ( member(Dz, D0), Dz = disease(Dx), causes(Dx, S) ->
        D = D0
    ; causes(Dx, S),
      prove(disease(Dx), D0, D, K)
    ).

% Non-abducible: prove from program
prove(G, D, D, _) :-
    \+ is_abducible(G),
    call(G), !.

% Abduce an abducible literal if budget allows and ICs not violated
prove(G, D0, D, K) :-
    is_abducible(G),
    K > 0,
    \+ memberchk_eq(G, D0),
    K1 is K - 1,
    D1 = [G|D0],
    \+ violates_ic(D1),
    D = D1.

is_abducible(G) :- functor(G, F, N), abducible(F/N).
memberchk_eq(X, [Y|_]) :- X =@= Y, !.
memberchk_eq(X, [_|T]) :- memberchk_eq(X, T).

% ------------------------ Integrity constraints ------------------------

add_ic(Pat) :- must_be(list, Pat), assertz(ic(Pat)).
clear_ics :- retractall(ic(_)).

valid_ic(D) :- \+ violates_ic(D).
violates_ic(D) :- ic(P), subset_match(P, D), !.

subset_match([], _).
subset_match([P|Ps], D) :-
    member(E, D),
    P =@= E,
    subset_match(Ps, D).

% ------------------------ Observation helpers ------------------------

assume_obs(ObsList) :- maplist(assertz_obs, ObsList).
assertz_obs(S) :- (obs(S) -> true ; assertz(obs(S))).
clear_obs :- retractall(obs(_)).

% ------------------------ Iterative deepening ------------------------
between(L, _, L).
between(L, inf, X) :- succ(L, L1), between(L1, inf, X).

% ------------------------ Diagnosis KB: Gout vs RA ------------------------

% Only diseases are abducible
:- initialization(init_abducibles).
init_abducibles :-
    retractall(abducible(_)),
    assertz(abducible(disease/1)).

% Gout (acute inflammatory, often monoarticular)
causes(gout, monoarticular_big_toe_pain).
causes(gout, sudden_onset_night).
causes(gout, severe_joint_pain).
causes(gout, redness).
causes(gout, swelling).
causes(gout, warmth).
causes(gout, uric_acid_high).
causes(gout, response_to_colchicine).

% Rheumatoid arthritis (chronic, symmetric small joints)
causes(rheumatoid_arthritis, symmetric_small_joints_pain).
causes(rheumatoid_arthritis, morning_stiffness_gt_60min).
causes(rheumatoid_arthritis, swelling).
causes(rheumatoid_arthritis, elevated_esr_crp).
causes(rheumatoid_arthritis, rheumatoid_factor_positive).
causes(rheumatoid_arthritis, anti_ccp_positive).
causes(rheumatoid_arthritis, chronic_course).

% Integrity constraints (optional): forbid co-abducing gout and RA
:- initialization(init_demo_ics).
init_demo_ics :-
    clear_ics,
    add_ic([disease(gout), disease(rheumatoid_arthritis)]).

% Convenience: reset observations + ICs
load_demo :-
    clear_obs,
    init_demo_ics.

% ---------- LP consistency ----------

% lp_consistent(+HList)
% True if HList is consistent with the KB (no contradictions).
lp_consistent(HList) :-
    % Example: just assume all are consistent for now
    % Replace with your real LP checks (e.g., no conflicts, no negation).
    \+ inconsistent(HList).

% You could define `inconsistent/1` in terms of your rules if needed.

% ---------- PLP probability ----------

% plp_prob(+HList, -P)
% Returns joint probability P(HList).
plp_prob([], 1.0).
plp_prob([H|T], P) :-
    plp_prob_single(H, PH),
    plp_prob(T, PT),
    P is PH * PT.

% plp_prob_single(+H, -P)
% Define probabilities for each hypothesis literal (toy example).
plp_prob_single(disease(gout), 0.3).
plp_prob_single(disease(rheumatoid_arthritis), 0.2).
plp_prob_single(rare_genetic_factor(_), 0.01).
plp_prob_single(_, 0.1).

% ---------- Argumentation defeat ----------

% defeat_strength(+Claim, -D)
% D in [0,1], 0 = undefeated, 1 = fully defeated.
% Here we just stub it; integrate your argumentation engine instead.
defeat_strength(Claim, 0.0) :-
    % if no attacking argument found:
    \+ attacked(Claim), !.
defeat_strength(Claim, 0.7) :-
    attacked(Claim), !.

% attacked/1 can be defined by your argumentation framework.

% ---------- Abduction cost ----------

% abduction_cost(+HList, -Cost)
% Basic MDL-like cost: number of hypotheses or more complex formula.
abduction_cost(HList, Cost) :-
    length(HList, N),
    Cost is N.

% ---------- Conditional probability P(EDU|H) ----------

% p_edu_given_h(+ClaimAtom, +HList, -P)
% Probability that ClaimAtom holds given HList.
p_edu_given_h(Claim, HList, 0.9) :-
    % Example: if Claim is entailed by HList, give high probability
    entails(HList, Claim), !.
p_edu_given_h(_, _, 0.1).

% entails/2 must be defined using your ALP / LP rules.