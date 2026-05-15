/* ============================================================
   LLP DEMO: Labeled Logic Program with full G implementation
   Single-file runnable example
   ============================================================ */

:- dynamic fact/2.
:- dynamic rule/3.

% ============================================================
% LABEL STRUCTURE
% lab(Time, Space, World, Source, Argument, Probability, Resource)
% ============================================================

% Example:
% lab(time(1,2), none, w0, src(xray,0.8),
%     arg(support,1,[xray],[]), prob(0.6,0.8), none)

% ============================================================
% G COMPONENTS
% ============================================================

% ---------- Priority (<) ----------
stronger(arg(_,P1,_,_), arg(_,P2,_,_)) :- P1 > P2.

% ---------- Compatibility F ----------
compatible(lab(T1,_,W1,_,_,_,_), lab(T2,_,W2,_,_,_,_)) :-
    time_compatible(T1,T2),
    world_compatible(W1,W2).

time_compatible(none,_).
time_compatible(_,none).
time_compatible(time(F1,T1), time(F2,T2)) :-
    max(F1,F2) =< min(T1,T2).

world_compatible(none,_).
world_compatible(_,none).
world_compatible(W,W).

% ---------- Propagation f ----------
combine(lab(T1,S1,W1,src(A,R1),Arg1,prob(L1,H1),_),
        lab(T2,S2,W2,src(B,R2),Arg2,prob(L2,H2),_),
        lab(T,S,W,src([A,B],R),Arg,prob(L,H),none)) :-

    % Time intersection
    combine_time(T1,T2,T),

    % Space/world (simple)
    (S1 = none -> S=S2 ; S=S1),
    (W1 = none -> W=W2 ; W=W1),

    % Source reliability multiplication
    R is R1 * R2,

    % Argument combine
    combine_arg(Arg1, Arg2, Arg),

    % Probability intersection
    L is max(L1,L2),
    H is min(H1,H2).

combine_time(none,X,X).
combine_time(X,none,X).
combine_time(time(F1,T1), time(F2,T2), time(F,T)) :-
    F is max(F1,F2),
    T is min(T1,T2).

combine_arg(arg(R1,P1,S1,A1), arg(R2,P2,S2,A2),
            arg(R,P,S,A)) :-
    P is max(P1,P2),
    append(S1,S2,S0), sort(S0,S),
    append(A1,A2,A0), sort(A0,A),
    role_merge(R1,R2,R).

role_merge(exception,_,exception).
role_merge(_,exception,exception).
role_merge(attack,_,attack).
role_merge(_,attack,attack).
role_merge(support,support,support).
role_merge(_,_,neutral).

% ---------- Aggregation ⊕ ----------
aggregate(L1,L2,Lout) :-
    L1 = lab(_,_,_,_,Arg1,_,_),
    L2 = lab(_,_,_,_,Arg2,_,_),
    ( stronger(Arg2,Arg1) ->
        Lout = L2
    ;
        Lout = L1
    ).

% ============================================================
% PROOF ENGINE
% ============================================================

prove(Atom, Label) :-
    fact(Label, Atom).

prove(Atom, LabelOut) :-
    rule(Lr, Atom, Body),
    prove_body(Body, LabelBody),
    compatible(Lr, LabelBody),
    combine(Lr, LabelBody, LabelOut).

prove_body([pos(A)], L) :-
    prove(A, L).

prove_body([pos(A)|Rest], Lout) :-
    prove(A, L1),
    prove_body(Rest, L2),
    combine(L1, L2, Lout).

% ============================================================
% DEFEAT CHECK
% ============================================================

defeated(lab(_,_,_,_,arg(exception,P,_,_),_,_)) :-
    P >= 2.

% ============================================================
% DEMO KNOWLEDGE BASE
% ============================================================

init_kb :-
    retractall(fact(_,_)),
    retractall(rule(_,_,_)),

    % FACTS
    assertz(fact(
      lab(time(1,2), none, w0, src(xray,0.8),
          arg(support,1,[xray],[]), prob(0.6,0.8), none),
      chest_xray_infiltrate(john)
    )),

    assertz(fact(
      lab(time(1,2), none, w0, src(lab,0.9),
          arg(exception,2,[viral_test],[]), prob(0.7,0.9), none),
      viral_infection(john)
    )),

    % RULES
    assertz(rule(
      lab(none,none,w0,src(guideline,0.7),
          arg(support,1,[r1],[]), prob(0.5,0.7), none),
      diagnosis(john,pneumonia),
      [pos(chest_xray_infiltrate(john))]
    )),

    assertz(rule(
      lab(none,none,w0,src(guideline,0.9),
          arg(exception,2,[r2],[]), prob(0.6,0.8), none),
      diagnosis(john,pneumonia),
      [pos(viral_infection(john))]
    )).

% ============================================================
% DEMO RUN
% ============================================================

run_demo :-
    init_kb,
    findall(L, prove(diagnosis(john,pneumonia), L), Labels),
    write('All derivations:'), nl,
    print_list(Labels),

    aggregate_all(Labels, Final),
    nl, write('Final aggregated label:'), nl,
    writeln(Final),

    ( defeated(Final)
    -> writeln('RESULT: Diagnosis is DEFEATED')
    ;  writeln('RESULT: Diagnosis is SUPPORTED')
    ).

% helper
aggregate_all([L],L).
aggregate_all([L1,L2|Rest],Lout) :-
    aggregate(L1,L2,L3),
    aggregate_all([L3|Rest],Lout).

print_list([]).
print_list([H|T]) :-
    writeln(H),
    print_list(T).

% ============================================================
% RUN:
% ?- run_demo.
% ============================================================