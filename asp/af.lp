arg(a). arg(b). arg(c).
att(a,b). att(b,c). att(c,a).

{ in(X) : arg(X) }.

:- in(X), in(Y), att(X,Y).
attacked(X) :- in(Y), att(Y,X).
:- arg(X), not in(X), not attacked(X).

#show in/1.
