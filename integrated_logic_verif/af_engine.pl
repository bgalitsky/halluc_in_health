:- module(af_engine, [
    grounded_extension/1,      % grounded_extension(-Extension)
    in_grounded/1,             % in_grounded(+Arg)
    acceptable_wrt/2           % acceptable_wrt(+Arg, +Extension)
]).

/*  Simple Dung-style Abstract Argumentation meta-interpreter.

    - arg(A).        declares an argument
    - attacks(A,B).  A attacks B

    Grounded extension is computed as the least fixed point of the
    characteristic function.
*/

:- use_module(library(lists)).

grounded_extension(Extension) :-
    grounded_fixpoint([], Extension).

grounded_fixpoint(Old, Old) :-
    next_grounded(Old, New),
    same_set(Old, New), !.
grounded_fixpoint(Old, Extension) :-
    next_grounded(Old, New),
    grounded_fixpoint(New, Extension).

% next_grounded(CurrentExt, NewExt)
next_grounded(Current, New) :-
    findall(A, (arg(A), acceptable_wrt(A, Current)), Ext),
    sort(Ext, New).

% Arg acceptable wrt Extension if all its attackers are attacked by Extension.
acceptable_wrt(A, Ext) :-
    findall(B, attacks(B, A), Attackers),
    forall(member(B, Attackers),
           ( member(C, Ext), attacks(C, B) )).

% Convenience: simply query membership in grounded extension
in_grounded(A) :-
    grounded_extension(Ext),
    member(A, Ext).

same_set(A, B) :-
    sort(A, SA),
    sort(B, SB),
    SA == SB.
