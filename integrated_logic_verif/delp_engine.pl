:- module(delp_engine, [
    strict_derivable/2,      % strict_derivable(+Lit, -UsedRules)
    defeasible_argument/3,   % defeasible_argument(+Lit, -Rules, -Type)
    defeats/2,               % defeats(+Arg1, +Arg2)
    warranted/1              % warranted(+Lit)
]).

/*  Simplified DeLP-style Defeasible Logic Programming engine.

    Representation:
      - fact(Lit).
      - strict_rule(Head, BodyList).
      - defeasible_rule(Id, Head, BodyList).
      - conflict(Lit, NegLit).
      - preferred_over(RuleId1, RuleId2).  % RuleId1 is preferred to RuleId2

    An "argument" is represented as arg(Lit, Rules, Type)
      where Type ∈ {strict, defeasible}.
*/

:- use_module(library(lists)).

%% strict_derivable(+Lit, -UsedRules)
strict_derivable(Lit, []) :-
    fact(Lit), !.

strict_derivable(Lit, [strict_rule(Lit, Body)|Used]) :-
    strict_rule(Lit, Body),
    strict_body(Body, Used).

strict_body([], []) :- !.
strict_body([L|Ls], Used) :-
    strict_derivable(L, Used1),
    strict_body(Ls, Used2),
    append(Used1, Used2, Used0),
    sort(Used0, Used).

%% defeasible_argument(+Lit, -Rules, -Type)
%  Type = strict if purely strict, defeasible otherwise.
defeasible_argument(Lit, Rules, strict) :-
    strict_derivable(Lit, Rules), !.

defeasible_argument(Lit, Rules, defeasible) :-
    defeasible_rule(Id, Lit, Body),
    defeasible_body(Body, RulesBody),
    Rules0 = [defeasible_rule(Id, Lit, Body)|RulesBody],
    sort(Rules0, Rules).

defeasible_body([], []) :- !.
defeasible_body([L|Ls], Rules) :-
    defeasible_argument(L, R1, _),
    defeasible_body(Ls, R2),
    append(R1, R2, R0),
    sort(R0, Rules).

%% defeats(+Arg1, +Arg2)
%  Very simplified: Arg1 defeats Arg2 if
%    - they support conflicting literals, and
%    - Arg1 is not strictly worse by priorities.
defeats(arg(L1, Rules1, _Type1), arg(L2, Rules2, _Type2)) :-
    conflict(L1, L2),
    \+ strictly_worse(Rules1, Rules2).

strictly_worse(R1, R2) :-
    member(defeasible_rule(Id1,_,_), R1),
    member(defeasible_rule(Id2,_,_), R2),
    preferred_over(Id2, Id1).  % Arg1 uses a rule worse than some in Arg2.

%% warranted(+Lit)
%  There exists an argument Arg for Lit that is not defeated by any argument for a conflict.
warranted(Lit) :-
    defeasible_argument(Lit, Rules, Type),
    Arg = arg(Lit, Rules, Type),
    \+ exists_defeater(Arg).

exists_defeater(Arg) :-
    Arg = arg(Lit, _, _),
    conflict(Lit, Opp),
    defeasible_argument(Opp, RulesOpp, TypeOpp),
    OppArg = arg(Opp, RulesOpp, TypeOpp),
    defeats(OppArg, Arg).
