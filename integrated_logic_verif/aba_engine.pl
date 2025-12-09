:- module(aba_engine, [
    derive/3,             % derive(+Goal, -Assumptions, +Visited)
    support_set/2,        % support_set(+Goal, -AssSet)
    attack/2,             % attack(+AssSet1, +AssSet2)
    admissible/1,         % admissible(+AssSet)
    supported_in_ABA/1    % supported_in_ABA(+Goal)
]).

/*  Simple Assumption-Based Argumentation engine.

    Representation:
      - rule(Head, BodyLits).
      - assumption(A).
      - contrary(A, C).

    Idea:
      - derive/3 constructs a proof for a literal and collects assumptions.
      - attack(Δ1, Δ2) holds if Δ1 can derive a contrary of any assumption in Δ2.
      - admissible(Δ) is a rough check: no internal attack and defends vs attackers.
*/

:- use_module(library(lists)).

%% derive(+Goal, -Assumptions, +Visited)
%  Visited is to avoid loops in rule applications.
derive(true, [], _) :- !.
derive((A,B), Assumptions, Visited) :- !,
    derive(A, Ass1, Visited),
    derive(B, Ass2, Visited),
    append(Ass1, Ass2, Assumptions0),
    sort(Assumptions0, Assumptions).

% Assumption used as fact
derive(Goal, [Goal], _) :-
    assumption(Goal), !.

% Apply rule(Head, BodyList)
derive(Goal, Assumptions, Visited) :-
    \+ member(Goal, Visited),
    rule(Goal, BodyList),
    derive_body_list(BodyList, Assumptions, [Goal|Visited]).

derive_body_list([], [], _) :- !.
derive_body_list([L|Ls], Assumptions, Visited) :-
    derive(L, A1, Visited),
    derive_body_list(Ls, A2, Visited),
    append(A1, A2, A0),
    sort(A0, Assumptions).

%% support_set(+Goal, -AssSet)
support_set(Goal, AssSet) :-
    derive(Goal, AssSet0, []),
    sort(AssSet0, AssSet).

%% attack(+AssSet1, +AssSet2)
%  Δ1 attacks Δ2 if Δ1 can derive a contrary of some assumption in Δ2.
attack(Ass1, Ass2) :-
    member(A2, Ass2),
    contrary(A2, C),
    derive(C, AssC, []),
    subset(AssC, Ass1).

%% admissible(+AssSet)
%  Rough admissibility: no self-attack, and defends against all attacks.
admissible(AssSet) :-
    \+ attack(AssSet, AssSet),              % no self-attack
    \+ exists_undefended_attack(AssSet).

exists_undefended_attack(AssSet) :-
    other_assumptions(OtherSet),
    attack(OtherSet, AssSet),
    \+ attack(AssSet, OtherSet).

other_assumptions(Other) :-
    findall(A, assumption(A), All),
    subset(Other, All),
    Other \== [].

%% supported_in_ABA(+Goal)
%  There exists an admissible set of assumptions that supports Goal.
supported_in_ABA(Goal) :-
    support_set(Goal, AssSet),
    admissible(AssSet).

subset([], _).
subset([X|Xs], set) :- member(X, set), subset(Xs, set).
