- module(abductive_solver, [
    validate_candidate_constraints/2,
    detect_cyclic_dependency_loop/2,
    check_boundary_invariants/3,
    run_abductive_repair/4
]).

/** <module> abductive_solver
 * Explicit Constraint and Inductive-Abductive Repair Engine
 * Framework Context: 2026 Computational Invention Logic Suite
 *
 * This module enforces strict hard constraints, loops through cyclic causal
 * topologies, maps structural boundaries, and proposes repairs for artifacts.
 */

% ==============================================================================
% 1. CORE DECLARATIVE DICTIONARY & COMPONENT DEFINITIONS
% ==============================================================================
is_base_function(a).
is_required_interface(b).
is_incompatible_option_c(c).
is_incompatible_option_d(d).
is_enhancement_node(e).

% Causal topology links: component_link(Source, Target)
component_link(c, d) :- fail. % Disallowed structural topology
component_link(e, b).        % Invariant: e strictly requires b to function
component_link(b, a).        % Interface feeds core functional loops

% ==============================================================================
% 2. DEEP STRUCTURAL LOOP AND CAUSAL PATH DETECTION
% ==============================================================================
% Detects path existence while avoiding infinite backtracking loops via Visited arrays
has_causal_path(X, Y, _) :-
    component_link(X, Y).
has_causal_path(X, Y, Visited) :-
    component_link(X, Z),
    \+ member(Z, Visited),
    has_causal_path(Z, Y, [Z|Visited]).

% Public interface to search for invalid functional feedback loops
detect_cyclic_dependency_loop(StartNode, LoopPath) :-
    has_causal_path(StartNode, StartNode, [StartNode]),
    LoopPath = [StartNode, StartNode].
detect_cyclic_dependency_loop(StartNode, [StartNode, NextNode | PathTail]) :-
    component_link(StartNode, NextNode),
    has_causal_path(NextNode, StartNode, [StartNode]),
    LoopPathTail = [NextNode | PathTail],
    find_loop_trace(NextNode, StartNode, [StartNode], LoopPathTail).

find_loop_trace(Target, Target, _, [Target]).
find_loop_trace(Current, Target, Visited, [Current | Rest]) :-
    component_link(Current, Next),
    \+ member(Next, Visited),
    find_loop_trace(Next, Target, [Next|Visited], Rest).

% ==============================================================================
% 3. BOUNDARY INVARIANTS AND PARAMETER ENFORCEMENT
% ==============================================================================
% Verifies parameter thresholds remain bounded within the stable window tau_S [1.5, 2.5]
check_boundary_invariants(Tau_S, _, Status) :-
    Tau_S < 1.5, !,
    Status = failure(tau_s_below_minimum_threshold).
check_boundary_invariants(Tau_S, _, Status) :-
    Tau_S > 2.5, !,
    Status = failure(tau_s_exceeds_maximum_allowable_bound).
check_boundary_invariants(_, CandidateAttributes, Status) :-
    member(c, CandidateAttributes),
    member(d, CandidateAttributes), !,
    Status = failure(hard_exclusion_violation_c_and_d).
check_boundary_invariants(_, CandidateAttributes, Status) :-
    member(e, CandidateAttributes),
    \+ member(b, CandidateAttributes), !,
    Status = failure(implication_violation_enhancement_requires_interface).
check_boundary_invariants(_, _, Status) :-
    Status = pass.

% ==============================================================================
% 4. COMPREHENSIVE MULTI-LAYERED GATEWAY EVALUATION
% ==============================================================================
validate_candidate_constraints(CandidateAttributes, Result) :-
    % 1. Evaluate hard attribute collisions
    check_boundary_invariants(2.0, CandidateAttributes, BoundaryStatus),
    (BoundaryStatus \= pass -> Result = BoundaryStatus ;

    % 2. Trace underlying causal links to catch circular structural faults
    (extract_candidate_links(CandidateAttributes, StructuralLinks),
     check_links_for_cycles(StructuralLinks, CyclicNode) ->
        Result = failure(cyclic_feedback_loop_detected(CyclicNode)) ;

    % 3. Confirm target functional coverage
    (member(a, CandidateAttributes), member(b, CandidateAttributes) ->
        Result = pass ;
        Result = failure(insufficient_target_functional_coverage)
    )).

extract_candidate_links(Attributes, FoundLinks) :-
    findall(link(X, Y), (member(X, Attributes), member(Y, Attributes), component_link(X, Y)), FoundLinks).

check_links_for_cycles(Links, FaultyNode) :-
    member(link(X, Y), Links),
    has_causal_path(Y, X, [X]).

% ==============================================================================
% 5. ABDUCTIVE MECHANISTIC REPAIR MATRIX
% ==============================================================================
run_abductive_repair(InitialAttributes, Tau_S, RepairedAttributes, ActionLog) :-
    check_boundary_invariants(Tau_S, InitialAttributes, Status),
    resolve_faults(Status, InitialAttributes, RepairedAttributes, ActionLog).

resolve_faults(pass, Attributes, Attributes, [no_action_required_candidate_stable]).
resolve_faults(failure(hard_exclusion_violation_c_and_d), Initial, Repaired, [dropped_mutually_exclusive_attribute_d | Log]) :-
    select(d, Initial, ResidualSet),
    check_boundary_invariants(2.0, ResidualSet, NextStatus),
    resolve_faults(NextStatus, ResidualSet, Repaired, Log).
resolve_faults(failure(implication_violation_enhancement_requires_interface), Initial, Repaired, [appended_missing_dependent_interface_b | Log]) :-
    UpdatedSet = [b | Initial],
    check_boundary_invariants(2.0, UpdatedSet, NextStatus),
    resolve_faults(NextStatus, UpdatedSet, Repaired, Log).
resolve_faults(failure(insufficient_target_functional_coverage), Initial, Repaired, [asserted_missing_core_functional_base_a | Log]) :-
    ( \+ member(a, Initial) -> NextSet = [a | Initial] ; NextSet = Initial ),
    ( \+ member(b, NextSet) -> TargetSet = [b | NextSet] ; TargetSet = NextSet ),
    validate_candidate_constraints(TargetSet, pass),
    Repaired = TargetSet,
    Log = [repair_gateway_promotion_certified].


:- module(abductive_solver, [
    validate_candidate/2,
    abductive_repair/4
]).

% Hard Constraint specifications
% target_passes(+Intent)
target_passes(Intent) :-
    member(a, Intent),
    member(b, Intent).

% integrity_constraints_passed(+Intent)
integrity_constraints_passed(Intent) :-
    \+ (member(c, Intent), member(d, Intent)), % \+ (c /\ d)
    (member(e, Intent) -> member(b, Intent) ; true). % e -> b

% validate_candidate(+Intent, -Status)
validate_candidate(Intent, pass) :-
    target_passes(Intent),
    integrity_constraints_passed(Intent), !.
validate_candidate(_, fail).

% core_preserving_residual_search(+Core, +AvailableResiduals, -ChosenResidual)
core_preserving_residual_search(Core, AvailableResiduals, ChosenResidual) :-
    include(is_feasible_residual(Core), AvailableResiduals, FeasibleResiduals),
    sort_residuals_by_cost(Core, FeasibleResiduals, [ChosenResidual|_]).

is_feasible_residual(Core, Residual) :-
    append(Core, Residual, Combined),
    validate_candidate(Combined, pass).

% sort_residuals_by_cost(+Core, +Residuals, -Sorted)
sort_residuals_by_cost(Core, Residuals, Sorted) :-
    map_list_to_pairs(residual_cost(Core), Residuals, Pairs),
    keysort(Pairs, SortedPairs),
    pairs_values(SortedPairs, Sorted).

residual_cost(_Core, Residual, Cost) :-
    % Unit edit distance penalty for elements in Residual, minus weight for optimization fields
    length(Residual, Len),
    (member(e, Residual) -> Penalty is Len - 2 ; Penalty is Len),
    Cost is Penalty.

% abductive_repair(+Core, +InitialResidual, +UniverseAttributes, -RepairedCandidate)
abductive_repair(Core, _InitialResidual, UniverseAttributes, RepairedCandidate) :-
    % Subtract Core attributes from the Universe of valid attributes to isolate residual territory
    subtract(UniverseAttributes, Core, RemainderUniverse),
    % Generate sub-combinations of the remainder universe
    subsets(RemainderUniverse, ListOfResiduals),
    core_preserving_residual_search(Core, ListOfResiduals, BestResidual),
    append(Core, BestResidual, RepairedCandidate).

subsets([], [[]]).
subsets([H|T], Sub) :-
    subsets(T, Sub1),
    maplist(atom_concat_list(H), Sub1, Sub2),
    append(Sub1, Sub2, Sub).

atom_concat_list(H, Element, [H|Element]).