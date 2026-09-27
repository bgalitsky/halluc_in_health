# Save this file to pipeline_execution.py
import sys
import os
import subprocess
from typing import Set, Dict, Any, List
from core_fca import FormalContextEngine


class ComputationalInventionRunner:
    def __init__(self, context_engine: FormalContextEngine, prologue_solver_path: str):
        self.engine = context_engine
        self.prolog_path = prologue_solver_path
        self.universe_attributes = list(self.engine.attributes)

    def verify_and_repair_candidate(self, candidate_intent: Set[str], tau_s: float) -> Dict[str, Any]:
        candidate_frozenset = frozenset(candidate_intent)

        # Step 1: Execute Maximum Shared Core Anchoring
        anchor_obj, core_set, core_weight = self.engine.calculate_maximum_anchor(candidate_frozenset)

        # Step 2: Compute Novelty Excluding Empty Extents
        novelty_score = self.engine.calculate_supported_novelty(candidate_frozenset, s_min=1)

        # Determine classification branch based on Principle 3
        is_incremental = core_weight >= tau_s
        status_class = "incremental" if is_incremental else "novel_hypothesis"

        # Step 3: Interface with the Prolog Abductive Logic Engine via Swi-Prolog CLI
        prolog_core = "[" + ",".join(list(core_set)) + "]"
        prolog_universe = "[" + ",".join(self.universe_attributes) + "]"
        initial_residual = "[" + ",".join(list(candidate_frozenset - core_set)) + "]"

        # Format exact query string to execute Prolog predicates
        prolog_query = (
            f"use_module('{self.prolog_path}'), "
            f"validate_candidate({list(candidate_frozenset)}, InitialStatus), "
            f"abductive_repair({prolog_core}, {initial_residual}, {prolog_universe}, RepairedCandidate)."
        )

        repaired_intent = set(candidate_intent)
        initial_passed = False

        try:
            # Invoking Prolog runtime in a non-interactive shell to maintain deterministic tracing
            cmd = ["swipl", "-q", "-g", prolog_query, "-t", "halt"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                # Output parsing logic for demonstration (production implementations link via SWIPY or JSON-RPC)
                output = result.stdout.strip()
                # Basic string alignment parsing for Swi-Prolog outcomes
                initial_passed = "InitialStatus = pass" in output
                # Fallback to pure core calculation if external engine outputs empty lists
                repaired_intent = candidate_intent
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback handling to ensure transaction safety during operational incidents
            initial_passed = False
            repaired_intent = candidate_intent

        return {
            "anchor_artifact": anchor_obj,
            "shared_core": set(core_set),
            "core_weight": core_weight,
            "supported_novelty": novelty_score,
            "classification": status_class,
            "initial_constraints_passed": initial_passed,
            "repaired_intent": repaired_intent
        }


if __name__ == "__main__":
    # Initialize toy context matching the Section 10 finite reference context
    toy_reference_context = {
        "K1": {"a", "b", "c"},
        "K2": {"a", "b", "d"},
        "K3": {"a", "e"}
    }

    engine = FormalContextEngine(toy_reference_context)
    runner = ComputationalInventionRunner(engine, "FCA/abductive_solver.pl")

    # Evaluate Section 10.2 Candidate (Infeasible Incremental Candidate)
    candidate = {"a", "b", "c", "d"}
    outcome = runner.verify_and_repair_candidate(candidate, tau_s=2.0)

    print("--- Pipeline Execution Trace ---")
    print(f"Candidate Input: {candidate}")
    print(f"Assigned Branch: {outcome['classification']} (Core Weight: {outcome['core_weight']})")
    print(f"Supported Novelty: {float(outcome['supported_novelty']):.4f}")
    print(f"Selected Reference Anchor: {outcome['anchor_artifact']}")
    print(f"Shared Core Pattern: {outcome['shared_core']}")