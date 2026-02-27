%temp profile will be sensitive to time(From,To) compatibility (overlap/intersection).

%arg profile enables a (very simple) defeat check using arg(Role,Priority,...).

%prob profile intersects probability intervals along the proof.

%hybrid combines all of the above.

/*  ============================================================
    Labeled Logic Program (LLP) in Prolog (SWI-Prolog friendly)
    ------------------------------------------------------------
    Goal: A single labeled logic program that can be "run" under
    different logical formalisms (temporal, modal, argumentation,
    probabilistic, provenance) by switching a PROFILE.

    - Facts:   fact(Label, Atom).
    - Rules:   rule(Label, Head, BodyLits).  BodyLits are pos(A) or naf(A).
    - Derive:  prove(Profile, Atom, OutLabel).

    Labels are records:
      lab(Time, Space, World, Src, Arg, Prob, Res)

    This is intentionally lightweight and extensible.
    ============================================================ */

:- module(llp_labeled_lp, [
    set_profile/1,
    fact/2,
    rule/3,
    prove/3,
    explain/3,
    demo/0
]).

:- dynamic current_profile/1.
:- dynamic fact/2.
:- dynamic rule/3.

% ----------------------------
% Profiles (logic "views")
% ----------------------------

% Turn components on/off and choose policies.
profile(temp,  prof{use_time:true,  use_world:false, use_arg:false, use_prob:false}).
profile(modal, prof{use_time:false, use_world:true,  use_arg:false, use_prob:false}).
profile(arg,   prof{use_time:false, use_world:false, use_arg:true,  use_prob:false}).
profile(prob,  prof{use_time:false, use_world:false, use_arg:false, use_prob:true }).
profile(hybrid,prof{use_time:true,  use_world:true,  use_arg:true,  use_prob:true }).

set_profile(P) :-
    retractall(current_profile(_)),
    assertz(current_profile(P)).

% ----------------------------
% Label constructors/utilities
% ----------------------------

% Label layout: lab(Time, Space, World, Src, Arg, Prob, Res)
% - Time:  time(From,To) or none
% - Space: space(Name) or none
% - World: w(Name) or none
% - Src:   src(SourceId, Reliability0to1) or none
% - Arg:   arg(Role, Priority, SupportSet, AttackSet)
%          Role in {support, attack, exception, neutral}
% - Prob:  prob(Low,High) interval in [0,1]
% - Res:   res(Tokens) or none (resource/relevance placeholder)

label_none(lab(none, none, none, none, arg(neutral,0,[],[]), prob(1.0,1.0), none)).

% ----------------------------
% Core proof with labels
% ----------------------------

/* prove(ProfileName, Atom, LabelOut)
   - tries to derive Atom with a computed label
   - avoids loops via a visited set
*/
prove(ProfileName, Atom, LabelOut) :-
    profile(ProfileName, Prof),
    prove_(Prof, Atom, [], LabelOut).

prove_(Prof, Atom, Visited, LabelOut) :-
    (   memberchk(Atom, Visited)
    ->  fail
    ;   true
    ),
    % 1) Try direct fact
    (   fact(L0, Atom),
        apply_profile(Prof, L0, LabelOut)
    ;   % 2) Try rule
        rule(Lr0, Atom, Body),
        prove_body(Prof, Body, [Atom|Visited], BodyLabels),
        % Combine rule label with body labels => conclusion label
        combine_labels(Prof, Lr0, BodyLabels, L1),
        apply_profile(Prof, L1, LabelOut),
        % Optional: argumentation defeat check
        (   Prof.use_arg == true
        ->  \+ defeated(LabelOut)
        ;   true
        )
    ).

prove_body(_Prof, [], _Visited, []).
prove_body(Prof, [pos(A)|Rest], Visited, [LA|Ls]) :-
    prove_(Prof, A, Visited, LA),
    prove_body(Prof, Rest, Visited, Ls).
prove_body(Prof, [naf(A)|Rest], Visited, [Lnaf|Ls]) :-
    % Negation-as-failure: succeed if A is not provable
    (   prove_(Prof, A, Visited, _)
    ->  fail
    ;   % label for naf can record that it was assumed by failure
        label_for_naf(Prof, A, Lnaf),
        prove_body(Prof, Rest, Visited, Ls)
    ).

label_for_naf(_Prof, _A, lab(none,none,none,src(naf,0.6),arg(neutral,0,[],[]),prob(0.4,0.6),none)).

% ----------------------------
% Label compatibility + combine
% ----------------------------

/* combine_labels(Prof, RuleLabel, PremiseLabels, Out)
   This is the LDS "propagation discipline" M:
   - checks compatibility constraints
   - propagates label components
   - aggregates uncertainty, provenance, etc.
*/
combine_labels(Prof, Lr0, Premises, Out) :-
    % First, ensure compatibility under active dimensions
    compatible_all(Prof, Lr0, Premises),
    % Then fold combine2 across premises
    foldl(combine2(Prof), Premises, Lr0, Out0),
    % Normalize/cleanup if desired
    normalize_label(Out0, Out).

compatible_all(_Prof, _Lr, []).
compatible_all(Prof, Lr, [Lp|Ls]) :-
    compatible_pair(Prof, Lr, Lp),
    compatible_all(Prof, Lr, Ls).

compatible_pair(Prof, lab(T1,S1,W1,_,_,_,_), lab(T2,S2,W2,_,_,_,_)) :-
    ( Prof.use_time  == true  -> time_compatible(T1,T2)  ; true ),
    ( Prof.use_world == true  -> world_compatible(W1,W2) ; true ),
    % For space, we keep it permissive by default (can tighten later)
    ( S1 = none ; S2 = none ; S1 = S2 ),
    true.

time_compatible(none, _).
time_compatible(_, none).
time_compatible(time(F1,T1), time(F2,T2)) :-
    % Overlap (soft) compatibility. Replace with strict order if needed.
    MaxF is max(F1,F2),
    MinT is min(T1,T2),
    MaxF =< MinT.

world_compatible(none, _).
world_compatible(_, none).
world_compatible(w(W), w(W)).   % same world by default (can add accessibility)

% combine2: combine one premise label into accumulator
combine2(Prof, Lp, Acc, Out) :-
    Acc = lab(Ta,Sa,Wa,SrcA,ArgA,ProbA,ResA),
    Lp  = lab(Tp,Sp,Wp,SrcP,ArgP,ProbP,ResP),

    % Time propagation: intersect intervals when active
    ( Prof.use_time == true
    -> time_combine(Ta,Tp,Tout)
    ;  Tout = Ta
    ),

    % Space/world: keep if consistent
    space_combine(Sa,Sp,Sout),
    ( Prof.use_world == true
    -> world_combine(Wa,Wp,Wout)
    ;  Wout = Wa
    ),

    % Provenance: multiply reliabilities, keep "best" source id list-ish
    src_combine(SrcA, SrcP, SrcOut),

    % Argumentation metadata: merge supports/attacks; keep max priority
    arg_combine(ArgA, ArgP, ArgOut),

    % Probability: intersect intervals if prob profile active, else keep
    ( Prof.use_prob == true
    -> prob_combine(ProbA, ProbP, ProbOut)
    ;  ProbOut = ProbA
    ),

    % Resource: sum tokens if present
    res_combine(ResA, ResP, ResOut),

    Out = lab(Tout,Sout,Wout,SrcOut,ArgOut,ProbOut,ResOut).

time_combine(none, X, X).
time_combine(X, none, X).
time_combine(time(F1,T1), time(F2,T2), time(F,T)) :-
    F is max(F1,F2),
    T is min(T1,T2).

space_combine(none, X, X).
space_combine(X, none, X).
space_combine(space(A), space(A), space(A)).

world_combine(none, X, X).
world_combine(X, none, X).
world_combine(w(A), w(A), w(A)).

src_combine(none, X, X).
src_combine(X, none, X).
src_combine(src(Id1,R1), src(Id2,R2), src(ids([Id1,Id2]), R)) :-
    R is R1*R2.

arg_combine(arg(Role1,P1,S1,A1), arg(Role2,P2,S2,A2), arg(Role,P,S,A)) :-
    % role: if any is exception/attack, keep the "stronger" role marker
    role_merge(Role1, Role2, Role),
    P is max(P1,P2),
    append(S1,S2,S0), sort(S0,S),
    append(A1,A2,A0), sort(A0,A).

role_merge(exception, _, exception) :- !.
role_merge(_, exception, exception) :- !.
role_merge(attack, _, attack) :- !.
role_merge(_, attack, attack) :- !.
role_merge(support, support, support) :- !.
role_merge(R1, R2, R) :-
    % fallback
    (R1 \= neutral -> R = R1 ; R = R2).

prob_combine(prob(L1,H1), prob(L2,H2), prob(L,H)) :-
    L is max(L1,L2),
    H is min(H1,H2),
    L =< H.

res_combine(none, X, X).
res_combine(X, none, X).
res_combine(res(A), res(B), res(C)) :- C is A+B.

normalize_label(L, L).  % hook for future cleanup

% ----------------------------
% Profile application (projection)
% ----------------------------

/* apply_profile(Prof, LabelIn, LabelOut)
   - you can "project away" inactive dimensions to keep explanations clean
*/
apply_profile(Prof, lab(T,S,W,Src,Arg,Prob,Res), lab(T2,S2,W2,Src,Arg,Prob2,Res2)) :-
    (Prof.use_time  == true -> T2=T ; T2=none),
    S2=S, % keep space always for now
    (Prof.use_world == true -> W2=W ; W2=none),
    (Prof.use_prob  == true -> Prob2=Prob ; Prob2=prob(1.0,1.0)),
    Res2=Res.

% ----------------------------
% Argumentation defeat (simple)
% ----------------------------

/* defeated(Label)
   Very simple defeat criterion:
   - if label's Arg role is attack/exception with priority >= 1,
     treat it as defeating. In practice you'd compare to competing
     derivations (support vs attack) and use priorities/strengths.
*/
defeated(lab(_,_,_,_,arg(Role,P,_,_),_,_)) :-
    (Role == attack ; Role == exception),
    P >= 1.

% ----------------------------
% Explanation helper
% ----------------------------

explain(Profile, Atom, explanation{atom:Atom, profile:Profile, label:Label}) :-
    prove(Profile, Atom, Label).

% ----------------------------
% Demo knowledge base
% ----------------------------

/*
  Example: LLM claims "post-op immunosuppression caused pneumonia"
  We'll show:
  - temporal profile catches inconsistency (pneumonia before surgery)
  - argumentation profile can defeat via exception (viral exposure)
  - probabilistic profile reduces confidence
*/

demo :-
    retractall(fact(_,_)),
    retractall(rule(_,_,_)),
    set_profile(hybrid),
    load_demo_kb,
    forall(member(P,[temp,arg,prob,hybrid]),
           ( format("~n=== Profile: ~w ===~n",[P]),
             ( explain(P, caused_pneumonia, E)
             -> portray_clause(E)
             ;  format("No proof for caused_pneumonia under profile ~w~n",[P])
             ))).

load_demo_kb :-
    % Facts extracted from LLM + context
    % surgery at [10,12], pneumonia at [5,6] (before surgery) => temporal conflict
    assertz(fact(lab(time(10,12), none, w(w0), src(llm,0.7), arg(support,0,[surgery],[]), prob(0.6,0.8), none),
                 surgery(p1))),
    assertz(fact(lab(time(9,15), none, w(w0), src(llm,0.7), arg(support,0,[immunosuppressed],[]), prob(0.6,0.8), none),
                 immunosuppressed(p1))),
    assertz(fact(lab(time(5,6), none, w(w0), src(llm,0.7), arg(support,0,[pneumonia],[]), prob(0.6,0.8), none),
                 pneumonia(p1))),

    % Alternative explanation evidence (exception / attack)
    assertz(fact(lab(time(4,7), none, w(w0), src(chart,0.9), arg(exception,2,[viral_exposure],[]), prob(0.7,0.9), none),
                 viral_exposure(p1))),

    % Rules (domain knowledge), neutral object-level form
    % Rule: if surgery and immunosuppressed then post_op_state
    assertz(rule(lab(none,none,none,src(domain,0.95), arg(neutral,0,[],[]), prob(0.8,0.95), none),
                 post_op_state(X),
                 [pos(surgery(X)), pos(immunosuppressed(X))])),

    % Rule: post_op_state implies caused_pneumonia (LLM-like causal leap)
    assertz(rule(lab(none,none,none,src(domain,0.8), arg(support,0,[r_causal],[]), prob(0.5,0.8), none),
                 caused_pneumonia,
                 [pos(post_op_state(p1)), pos(pneumonia(p1))])),

    % Exception rule: viral exposure can explain pneumonia and defeat the causal link
    assertz(rule(lab(none,none,none,src(domain,0.9), arg(exception,2,[r_exception],[]), prob(0.7,0.9), none),
                 caused_pneumonia,
                 [pos(viral_exposure(p1)), pos(pneumonia(p1))])).
