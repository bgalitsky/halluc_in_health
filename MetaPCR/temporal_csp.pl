/*
   Temporal constraint satisfaction with SWI-Prolog CLP(FD).

   Event specifications:

       event(Id, Duration, ResourceUse)
       event(Id, range(MinDuration, MaxDuration), ResourceUse)

   Supported constraints:

       fixed_start(Id, Time)
       fixed_end(Id, Time)
       release(Id, EarliestStart)
       deadline(Id, LatestEnd)
       start_window(Id, Earliest, Latest)
       end_window(Id, Earliest, Latest)
       precedes(A, B, MinimumGap)
       gap(A, B, MinimumGap, MaximumGap)
       same_start(A, B)
       same_end(A, B)
       intersects(A, B)
       no_overlap(A, B)
       relation(Relation, A, B)
       resource_limit(Capacity)

   relation/3 supports the thirteen Allen interval relations:

       before, meets, overlaps, starts, during, finishes, equal,
       after, met_by, overlapped_by, started_by, contains, finished_by

   Time is discrete and intervals are interpreted as half-open [Start, End).

   Example SWI-Prolog session:

       ?- [temporal_csp].
       ?- example(Schedule), print_schedule(Schedule).
       ?- run_tests.
*/

:- module(temporal_csp,
          [ solve_schedule/3,
            solve_schedule/4,
            example/1,
            print_schedule/1
          ]).

:- use_module(library(clpfd)).
:- use_module(library(error)).
:- use_module(library(pairs)).


%!  solve_schedule(+Events, +Constraints, -Schedule) is semidet.
%
%   Solve within the default horizon 0..100 and minimize the makespan.

solve_schedule(Events, Constraints, Schedule) :-
    solve_schedule(Events, Constraints, 100, Schedule).


%!  solve_schedule(+Events, +Constraints, +Horizon, -Schedule) is semidet.
%
%   Schedule is a list of slot(Id, Start, End, Duration, ResourceUse).
%   A feasible schedule with minimum makespan is returned first.

solve_schedule(Events, Constraints, Horizon, Schedule) :-
    must_be(list, Events),
    must_be(list, Constraints),
    must_be(integer, Horizon),
    Horizon > 0,
    ensure_unique_ids(Events),
    maplist(build_slot(Horizon), Events, Schedule),
    maplist(post_constraint(Schedule), Constraints),
    Makespan in 0..Horizon,
    maplist(ends_by(Makespan), Schedule),
    term_variables([Makespan|Schedule], Variables),
    labeling([ffc, bisect, min(Makespan)], Variables).


build_slot(Horizon, event(Id, DurationSpec, ResourceUse),
           slot(Id, Start, End, Duration, ResourceUse)) :-
    !,
    must_be(atom, Id),
    must_be(integer, ResourceUse),
    ResourceUse >= 0,
    Start in 0..Horizon,
    End in 0..Horizon,
    constrain_duration(DurationSpec, Duration, Horizon),
    End #= Start + Duration.
build_slot(_, Invalid, _) :-
    domain_error(event_specification, Invalid).


constrain_duration(Duration0, Duration, Horizon) :-
    integer(Duration0),
    !,
    Duration0 > 0,
    Duration0 =< Horizon,
    Duration #= Duration0.
constrain_duration(range(Minimum, Maximum), Duration, Horizon) :-
    !,
    must_be(integer, Minimum),
    must_be(integer, Maximum),
    Minimum > 0,
    Minimum =< Maximum,
    Maximum =< Horizon,
    Duration in Minimum..Maximum.
constrain_duration(Invalid, _, _) :-
    domain_error(duration_specification, Invalid).


ensure_unique_ids(Events) :-
    maplist(event_id, Events, Ids),
    sort(Ids, UniqueIds),
    (   same_length(Ids, UniqueIds)
    ->  true
    ;   domain_error(unique_event_identifiers, Ids)
    ).


event_id(event(Id, _, _), Id) :-
    !,
    must_be(atom, Id).
event_id(Invalid, _) :-
    domain_error(event_specification, Invalid).


ends_by(Makespan, slot(_, _, End, _, _)) :-
    End #=< Makespan.


post_constraint(Slots, fixed_start(Id, Time)) :-
    !,
    slot_bounds(Id, Slots, Start, _),
    Start #= Time.
post_constraint(Slots, fixed_end(Id, Time)) :-
    !,
    slot_bounds(Id, Slots, _, End),
    End #= Time.
post_constraint(Slots, release(Id, EarliestStart)) :-
    !,
    slot_bounds(Id, Slots, Start, _),
    Start #>= EarliestStart.
post_constraint(Slots, deadline(Id, LatestEnd)) :-
    !,
    slot_bounds(Id, Slots, _, End),
    End #=< LatestEnd.
post_constraint(Slots, start_window(Id, Earliest, Latest)) :-
    !,
    slot_bounds(Id, Slots, Start, _),
    Start #>= Earliest,
    Start #=< Latest.
post_constraint(Slots, end_window(Id, Earliest, Latest)) :-
    !,
    slot_bounds(Id, Slots, _, End),
    End #>= Earliest,
    End #=< Latest.
post_constraint(Slots, precedes(A, B, MinimumGap)) :-
    !,
    must_be(integer, MinimumGap),
    MinimumGap >= 0,
    slot_bounds(A, Slots, _, EndA),
    slot_bounds(B, Slots, StartB, _),
    EndA + MinimumGap #=< StartB.
post_constraint(Slots, gap(A, B, MinimumGap, MaximumGap)) :-
    !,
    must_be(integer, MinimumGap),
    must_be(integer, MaximumGap),
    MinimumGap >= 0,
    MinimumGap =< MaximumGap,
    slot_bounds(A, Slots, _, EndA),
    slot_bounds(B, Slots, StartB, _),
    StartB - EndA #>= MinimumGap,
    StartB - EndA #=< MaximumGap.
post_constraint(Slots, same_start(A, B)) :-
    !,
    slot_bounds(A, Slots, StartA, _),
    slot_bounds(B, Slots, StartB, _),
    StartA #= StartB.
post_constraint(Slots, same_end(A, B)) :-
    !,
    slot_bounds(A, Slots, _, EndA),
    slot_bounds(B, Slots, _, EndB),
    EndA #= EndB.
post_constraint(Slots, intersects(A, B)) :-
    !,
    slot_bounds(A, Slots, StartA, EndA),
    slot_bounds(B, Slots, StartB, EndB),
    StartA #< EndB,
    StartB #< EndA.
post_constraint(Slots, no_overlap(A, B)) :-
    !,
    slot_bounds(A, Slots, StartA, EndA),
    slot_bounds(B, Slots, StartB, EndB),
    (EndA #=< StartB) #\/ (EndB #=< StartA).
post_constraint(Slots, relation(Relation, A, B)) :-
    !,
    interval_relation(Relation, A, B, Slots).
post_constraint(Slots, resource_limit(Capacity)) :-
    !,
    must_be(integer, Capacity),
    Capacity >= 0,
    maplist(slot_task, Slots, Tasks),
    cumulative(Tasks, [limit(Capacity)]).
post_constraint(_, Invalid) :-
    domain_error(temporal_constraint, Invalid).


slot_bounds(Id, Slots, Start, End) :-
    (   memberchk(slot(Id, Start, End, _, _), Slots)
    ->  true
    ;   existence_error(event, Id)
    ).


slot_task(slot(Id, Start, End, Duration, ResourceUse),
          task(Start, Duration, End, ResourceUse, Id)).


% Allen's thirteen basic interval relations.

interval_relation(before, A, B, Slots) :-
    !,
    slot_bounds(A, Slots, _, EndA),
    slot_bounds(B, Slots, StartB, _),
    EndA #< StartB.
interval_relation(meets, A, B, Slots) :-
    !,
    slot_bounds(A, Slots, _, EndA),
    slot_bounds(B, Slots, StartB, _),
    EndA #= StartB.
interval_relation(overlaps, A, B, Slots) :-
    !,
    slot_bounds(A, Slots, StartA, EndA),
    slot_bounds(B, Slots, StartB, EndB),
    StartA #< StartB,
    StartB #< EndA,
    EndA #< EndB.
interval_relation(starts, A, B, Slots) :-
    !,
    slot_bounds(A, Slots, StartA, EndA),
    slot_bounds(B, Slots, StartB, EndB),
    StartA #= StartB,
    EndA #< EndB.
interval_relation(during, A, B, Slots) :-
    !,
    slot_bounds(A, Slots, StartA, EndA),
    slot_bounds(B, Slots, StartB, EndB),
    StartB #< StartA,
    EndA #< EndB.
interval_relation(finishes, A, B, Slots) :-
    !,
    slot_bounds(A, Slots, StartA, EndA),
    slot_bounds(B, Slots, StartB, EndB),
    StartB #< StartA,
    EndA #= EndB.
interval_relation(equal, A, B, Slots) :-
    !,
    slot_bounds(A, Slots, StartA, EndA),
    slot_bounds(B, Slots, StartB, EndB),
    StartA #= StartB,
    EndA #= EndB.
interval_relation(after, A, B, Slots) :-
    !,
    interval_relation(before, B, A, Slots).
interval_relation(met_by, A, B, Slots) :-
    !,
    interval_relation(meets, B, A, Slots).
interval_relation(overlapped_by, A, B, Slots) :-
    !,
    interval_relation(overlaps, B, A, Slots).
interval_relation(started_by, A, B, Slots) :-
    !,
    interval_relation(starts, B, A, Slots).
interval_relation(contains, A, B, Slots) :-
    !,
    interval_relation(during, B, A, Slots).
interval_relation(finished_by, A, B, Slots) :-
    !,
    interval_relation(finishes, B, A, Slots).
interval_relation(Invalid, _, _, _) :-
    domain_error(allen_interval_relation, Invalid).


%!  example(-Schedule) is semidet.
%
%   A small MetaPCR-style verification pipeline. Validation and bridge
%   checking run concurrently without exceeding capacity 3.

example(Schedule) :-
    Events =
        [ event(collect_evidence, 3, 2),
          event(decompose,        2, 1),
          event(validate_logic,   4, 2),
          event(check_bridge,     2, 1),
          event(report,           1, 1)
        ],
    Constraints =
        [ fixed_start(collect_evidence, 0),
          precedes(collect_evidence, decompose, 0),
          precedes(decompose, validate_logic, 0),
          precedes(decompose, check_bridge, 0),
          same_start(validate_logic, check_bridge),
          precedes(validate_logic, report, 0),
          precedes(check_bridge, report, 0),
          deadline(report, 12),
          resource_limit(3)
        ],
    solve_schedule(Events, Constraints, 20, Schedule).


%!  print_schedule(+Schedule) is det.

print_schedule(Schedule) :-
    map_list_to_pairs(slot_sort_key, Schedule, Pairs),
    keysort(Pairs, OrderedPairs),
    pairs_values(OrderedPairs, Ordered),
    forall(member(slot(Id, Start, End, Duration, ResourceUse), Ordered),
           format('~w: ~d..~d (duration=~d, resource=~d)~n',
                  [Id, Start, End, Duration, ResourceUse])).


slot_sort_key(slot(Id, Start, _, _, _), Start-Id).


:- begin_tests(temporal_csp).

test(example_schedule,
     true(Schedule ==
          [ slot(collect_evidence, 0, 3, 3, 2),
            slot(decompose,        3, 5, 2, 1),
            slot(validate_logic,   5, 9, 4, 2),
            slot(check_bridge,     5, 7, 2, 1),
            slot(report,           9, 10, 1, 1)
          ])) :-
    example(Schedule).

test(infeasible_deadline, [fail]) :-
    solve_schedule([event(a, 5, 1)], [deadline(a, 4)], 10, _).

test(allen_overlap) :-
    Events = [event(a, 4, 1), event(b, 4, 1)],
    Constraints = [fixed_start(a, 0), relation(overlaps, a, b)],
    solve_schedule(Events, Constraints, 10, Schedule),
    memberchk(slot(b, StartB, _, _, _), Schedule),
    StartB >= 1,
    StartB =< 3.

:- end_tests(temporal_csp).
