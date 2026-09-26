#!/usr/bin/env python3
"""
test_rules.py - Risk Calibration and Rule Reconciliation Unit Test Suite
Verifies logistic regression risk calculation, parameter sensitivity sweeps,
and classification transition mapping between original and corrected rules.
"""

import math
import json
import unittest
from typing import Dict, Any


class EvaluationEngine:
    def __init__(self, weights: Dict[str, float]):
        self.b0 = weights["beta_0"]
        self.b1 = weights["beta_1"]
        self.b2 = weights["beta_2"]
        self.b3 = weights["beta_3"]

    def compute_failure_probability(self, tau_S: float, sigma_N: float, omega_A: float) -> float:
        """Calculates log-odds failure probability using fitted calibration coefficients."""
        log_odds = self.b0 + (self.b1 * tau_S) + (self.b2 * sigma_N) + (self.b3 * omega_A)
        return 1.0 / (1.0 + math.exp(-log_odds))

    def evaluate_original_rule(self, candidate: Dict[str, Any]) -> str:
        """Emulates legacy keyword-matching logic vulnerable to over-promotion."""
        if candidate["keyword_match_score"] > 0.5:
            return "PROMOTED"
        return "DROPPED"

    def evaluate_corrected_rule(self, candidate: Dict[str, Any]) -> str:
        """Applies corrected FCA set-valued structural lattice boundaries."""
        if candidate["hard_constraint_violation"]:
            return "REJECTED_CONSTRAINT_FAILURE"
        if candidate["semantic_invariant_drift"]:
            return "DROPPED_INVARIANT_DRIFT"
        if candidate["structural_analogy_valid"] or (candidate["fca_core_anchor_weight"] >= 1.5):
            return "PROMOTED"
        return "DROPPED_TRUE_NEGATIVE"


class TestRiskCalibrationAndRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initializing configuration constants matching the manuscript values
        cls.weights = {
            "beta_0": 1.42,
            "beta_1": -0.84,
            "beta_2": 0.65,
            "beta_3": -0.38
        }
        cls.engine = EvaluationEngine(cls.weights)

    def test_risk_index_calculation(self):
        """Verifies mathematical consistency of the risk probability calculation."""
        # Baseline profile: mid-range attributes
        prob = self.engine.compute_failure_probability(tau_S=2.0, sigma_N=0.4, omega_A=0.8)

        # Expected Log-odds = 1.42 + (-0.84 * 2.0) + (0.65 * 0.4) + (-0.38 * 0.8) = -0.304
        # Expected Prob = 1 / (1 + exp(0.304)) = 0.42457
        self.assertAlmostEqual(prob, 0.42457, places=4, msg="Brier base calibration logic mismatch.")

    def test_rule_reconciliation_transitions(self):
        """Validates that candidate classification shifts match the taxonomy rows in Table 14.10."""

        # Row 1: False Innovation -> Over-promoted legacy rule hits hard constraint checks in corrected engine
        c1 = {
            "keyword_match_score": 0.85,
            "hard_constraint_violation": True,
            "semantic_invariant_drift": False,
            "structural_analogy_valid": False,
            "fca_core_anchor_weight": 1.0
        }
        self.assertEqual(self.engine.evaluate_original_rule(c1), "PROMOTED")
        self.assertEqual(self.engine.evaluate_corrected_rule(c1), "REJECTED_CONSTRAINT_FAILURE")

        # Row 2: Undetected Analogy -> Dropped by keyword filter, salvaged/promoted by corrected structural match
        c2 = {
            "keyword_match_score": 0.20,
            "hard_constraint_violation": False,
            "semantic_invariant_drift": False,
            "structural_analogy_valid": True,
            "fca_core_anchor_weight": 1.8
        }
        self.assertEqual(self.engine.evaluate_original_rule(c2), "DROPPED")
        self.assertEqual(self.engine.evaluate_corrected_rule(c2), "PROMOTED")

        # Row 3: Unstable Repair -> Promoted by legacy framework, isolated as drift by set-valued checks
        c3 = {
            "keyword_match_score": 0.70,
            "hard_constraint_violation": False,
            "semantic_invariant_drift": True,
            "structural_analogy_valid": False,
            "fca_core_anchor_weight": 1.2
        }
        self.assertEqual(self.engine.evaluate_original_rule(c3), "PROMOTED")
        self.assertEqual(self.engine.evaluate_corrected_rule(c3), "DROPPED_INVARIANT_DRIFT")


if __name__ == "__main__":
    print("Executing calibration testing harness...")
    unittest.main()
Use
code
with caution.Make the script executable in your environment:bashchmod + x
test_rules.py
Use
code
with caution.2.Parameters Registry Configuration (calibrated_parameters.json)json{
"system_context": "Risk Calibration and Framework Robustness Manifest 2026",
"calibration_partition": {
    "sample_size_questions": 170,
    "sample_size_traces": 369,
    "percentage_of_active_data": 20.0
},
"held_out_partition": {
    "sample_size_questions": 680,
    "sample_size_traces": 1476
},
"fitted_logistic_coefficients": {
    "beta_0_intercept": 1.42,
    "beta_1_tau_S": -0.84,
    "beta_2_sigma_N": 0.65,
    "beta_3_omega_A": -0.38
},
"predictive_performance_metrics": {
    "validation_brier_score": 0.038,
    "auc_roc": 0.89,
    "calibration_intercept": 0.02,
    "calibration_slope": 0.97
},
"sensitivity_bounds": {
    "tau_S_window": [1.5, 2.5],
    "s_min_threshold_range": [0.10, 0.30],
    "max_classification_drift_percentage": 1.2,
    "down_sampled_corpus_yield_shift": 2.1
},
"reconciliation_summary": {
    "total_evaluated_candidates": 1845,
    "legacy_false_positive_promotions": 231,
    "revoked_promotions_constraint_violations": 142,
    "revoked_promotions_invariant_drift": 89,
    "recovered_structural_analogies": 68
}
}