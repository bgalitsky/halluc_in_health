:- module(alp_engine, [
    abduce/3,            % abduce(+Goal, -Delta, -Status)
    solve/3,             % solve(+Goal, +DeltaIn, -DeltaOut)
    check_ics/1          % check_ics(+Delta)
]).

/*  Abductive Logic Programming meta-interpreter (simple, educational version).

    - abduce/3: main entry point.
    - solve/3  : SLD-style proof with abduction of abducibles.
    - Delta    : list/set of abduced literals.
    - Integrity constraints ic(BodyList) must not be all simultaneously true.
*/

%% abduce(+Goal, -Delta, -Status)
%  Status ∈ {ok, ic_violated}.
abduce(Goal, Delta, Status) :-
    solve(Goal, [], Delta0),
    (   check_ics(Delta0)
    ->  Status = ok,
        Delta = Delta0
    ;   Status = ic_violated,
        Delta = Delta0
    ).

%% solve(+Goal, +DeltaIn, -DeltaOut)
solve(true, Delta, Delta) :- !.
solve((A,B), Delta0, Delta2) :- !,
    solve(A, Delta0, Delta1),
    solve(B, Delta1, Delta2).

% Defined predicate via program clauses
solve(Goal, Delta0, Delta) :-
    clause(Goal, Body),
    solve(Body, Delta0, Delta).

% Abducible literal
solve(Goal, Delta0, Delta) :-
    abducible(Goal),
    abduce_literal(Goal, Delta0, Delta).

% Built-in or ground literal not abducible: try directly
solve(Goal, Delta, Delta) :-
    \+ predicate_property(Goal, interpreted), % crude guard; adjust as needed
    call(Goal).

%% abduce_literal(+Lit, +DeltaIn, -DeltaOut)
abduce_literal(Lit, Delta, Delta) :-
    member(Lit, Delta), !.    % already assumed
abduce_literal(Lit, Delta0, [Lit|Delta0]) :-
    \+ inconsistent_abduction(Lit, Delta0).

%% inconsistent_abduction(+Lit, +Delta)
%  Inconsistent if there is contrary(Lit, Neg) and Neg derivable (or already abduced).
inconsistent_abduction(Lit, Delta) :-
    contrary(Lit, Neg),
    (   member(Neg, Delta)
    ;   solve(Neg, Delta, _)
    ), !.

%% check_ics(+Delta)
%  Integrity constraints ic(ListOfLits) must not all hold.
check_ics(Delta) :-
    \+ violates_ic(Delta).

violates_ic(Delta) :-
    ic(BodyLits),
    all_true(BodyLits, Delta), !.

all_true([], _) :- !.
all_true([L|Ls], Delta) :-
    (   member(L, Delta)
    ;   solve(L, Delta, _)
    ),
    all_true(Ls, Delta).
