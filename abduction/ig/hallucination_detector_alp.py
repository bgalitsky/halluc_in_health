# hallucination_detector_alp.py
from __future__ import annotations
import os
import sys
# Get parent directory of current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from abduction.alp_dx import AlpDiagnosis   # your ALP wrapper
import math

dx = AlpDiagnosis()
dx.load_demo()


# -------------------------
# Data structures
# -------------------------

@dataclass
class EDU:
    edu_id: str
    text: str
    weight: float                 # discourse weight w_i
    ig: float                     # information gain IG(c_i, S)
    symptoms: List[str]           # for ALP engine (obs)
    claim_atom: str               # main Prolog atom representing the claim
    label: Optional[int] = None   # gold label: 1 = hallucination, 0 = non-hallucination


@dataclass
class EDUDecision:
    edu: EDU
    hallucination: bool
    reason: str
    explanation: Optional[List[str]] = None
    score_claim: Optional[float] = None
    score_base: Optional[float] = None


# -------------------------
# IG Computer interface
# -------------------------

class IGComputer:
    """
    Pluggable IG computation.

    Implement compute_ig(edu, source) to:
      - call an NLI/LLM/QA auditor,
      - estimate KL divergence / IG for this EDU vs source.
    """

    def compute_ig(self, edu: EDU, source: str) -> float:
        """
        Default stub: if edu.ig already set, leave it.
        Otherwise, return 0.0.
        """
        return edu.ig if edu.ig is not None else 0.0


# -------------------------
# Hallucination Detector
# -------------------------

class IGAbductionHallucinationDetector:
    """
    Information-Gain + MDL-style scoring + Abduction + Counter-Abduction
    on top of AlpDiagnosis.
    """

    def __init__(
        self,
        alp: Optional[AlpDiagnosis] = None,
        ig_computer: Optional[IGComputer] = None,
        ig_low_threshold: float = 0.5,
        ig_high_threshold: float = 1.5,
        counter_margin: float = 0.1,
        alpha_h: float = 1.0,
        beta_residual: float = 1.0,
    ):
        """
        alpha_h: weight for hypothesis complexity L(H).
        beta_residual: weight for residual info (w_i * IG).
        """
        self.alp = alp if alp is not None else AlpDiagnosis()
        self.ig_computer = ig_computer if ig_computer is not None else IGComputer()
        self.ig_low = ig_low_threshold
        self.ig_high = ig_high_threshold
        self.margin = counter_margin
        self.alpha_h = alpha_h
        self.beta_residual = beta_residual

    # ----------------------------------
    # IG update step
    # ----------------------------------
    def update_ig(self, edu: EDU, source_text: str) -> None:
        """
        Compute or refine IG for an EDU given the source.
        """
        edu.ig = self.ig_computer.compute_ig(edu, source_text)

    # ----------------------------------
    # Abduction via ALP
    # ----------------------------------
    def abductive_explanation(self, edu: EDU) -> List[str]:
        """
        Uses your ALP engine to find abductive explanations for the EDU's symptoms.
        """
        if not edu.symptoms:
            return []

        try:
            # You can also use explain_obs_k or explain_obs_all if you want multiple hypotheses.
            explanation = self.alp.explain_obs(edu.symptoms)
            return explanation
        except Exception:
            return []

    # ----------------------------------
    # MDL-style scoring
    # ----------------------------------
    def score_hypothesis(self, edu: EDU, explanation: List[str]) -> float:
        """
        MDL-style score:
          Score(H) = alpha * L(H) + beta * w_i * IG(c_i, S)

        Where:
          - L(H) ~ length or complexity of explanation.
          - residual term approximated by discourse-weighted IG for this EDU.

        This is a simplification of:
           L(H) + Σ w_j L(EDU_j | H),
        restricted here to the focal EDU_i.
        """
        # Hypothesis complexity: each abduced literal gets unit cost
        L_H = self.alpha_h * float(len(explanation))

        # Residual cost: use discourse-weighted IG as a proxy
        L_residual = self.beta_residual * (edu.weight * edu.ig)

        return L_H + L_residual



    def logical_ensemble_score(dx: AlpDiagnosis, edu, explanation: list[str], source_text: str) -> float:
        """
        Full neuro-symbolic MDL-style ensemble score.

        Uses:
          - PLP:      L(H) = sum_h -log P(h)
          - Conditional: L(EDU|H) = -log P(edu | H)
          - Argumentation: defeat penalty
          - Discourse: edu.weight
          - Entropy:  edu.ig (information gain)

        Lower score = better (more plausible, less hallucinated).
        """

        # (1) Hypothesis description length L(H) via PLP
        L_H = 0.0
        for h in explanation:
            p_h = dx.plp_prob_single( h)
            L_H += -math.log(max(p_h, 1e-6))

        # (2) Conditional likelihood / residual cost L(EDU|H)
        p_edu_given_H = dx.conditional_entailment_prob( edu, explanation, source_text)
        L_residual = -math.log(max(p_edu_given_H, 1e-6))

        # Discourse weighting
        L_residual *= edu.weight

        # (3) Argumentation defeat penalty
        defeat_strength = dx.arg_defeat(edu.claim_atom)  # 0..1
        arg_cost = 2.0 * defeat_strength  # tune factor 2.0 as needed

        # (4) Information gain as entropy-like term
        info_entropy = edu.ig

        # (5) Optional: LP hard inconsistency penalty
        lp_ok = dx.lp_check( explanation)
        lp_cost = 0.0 if lp_ok == 1 else 5.0  # hard penalty if inconsistent

        # Final ensemble MDL score
        total_cost = L_H + L_residual + arg_cost + info_entropy + lp_cost
        return float(total_cost)

    def baseline_score(self, edu: EDU) -> float:
        """
        Baseline score approximating explanation *without* committing to the claim:
          here, use a minimal residual IG at the low threshold.

        You can refine this to include global context, other EDUs, etc.
        """
        return self.beta_residual * (edu.weight * self.ig_low)

    # ----------------------------------
    # Decision logic
    # ----------------------------------
    def classify_edu(self, edu: EDU, source_text: str) -> EDUDecision:
        """
        For a single EDU:
          1) Compute/refresh IG.
          2) If IG is low → classify as non-hallucination.
          3) If IG is high → call abduction and score.
          4) Apply counter-abduction criterion.
        """
        # Step 1: Update or confirm IG
        self.update_ig(edu, source_text)

        # Step 2: Low IG → highly likely consistent
        if edu.ig < self.ig_low:
            return EDUDecision(
                edu=edu,
                hallucination=False,
                reason=f"Low IG ({edu.ig:.3f}) — close to source-supported distribution.",
                explanation=None,
                score_claim=None,
                score_base=None,
            )

        # Step 3: Get abductive explanation
        explanation = self.abductive_explanation(edu)

        if not explanation:
            # No abductive explanation → strong counter-abductive failure
            return EDUDecision(
                edu=edu,
                hallucination=True,
                reason=(
                    f"IG={edu.ig:.3f} and no abductive explanation found "
                    f"(counter-abductive failure)."
                ),
                explanation=None,
                score_claim=None,
                score_base=None,
            )

        score_claim = self.score_hypothesis(edu, explanation)
        score_base = self.baseline_score(edu)

        # Step 4: Counter-abduction test
        if score_claim > score_base + self.margin:
            # Including the claim always makes coding worse → hallucination
            return EDUDecision(
                edu=edu,
                hallucination=True,
                reason=(
                    f"Counter-abductive failure: ScoreClaim={score_claim:.3f} "
                    f"> ScoreBase={score_base:.3f} + margin."
                ),
                explanation=explanation,
                score_claim=score_claim,
                score_base=score_base,
            )
        else:
            # Reasonable H exists that keeps score small → abduction success
            return EDUDecision(
                edu=edu,
                hallucination=False,
                reason=(
                    f"Abductively supported: ScoreClaim={score_claim:.3f} "
                    f"<= ScoreBase={score_base:.3f} + margin."
                ),
                explanation=explanation,
                score_claim=score_claim,
                score_base=score_base,
            )

    # ----------------------------------
    # Batch over EDUs
    # ----------------------------------
    def analyze_example(self, source_text: str, edus: List[EDU]) -> List[EDUDecision]:
        return [self.classify_edu(e, source_text) for e in edus]
