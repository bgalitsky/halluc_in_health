# hallucination_detector_med.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from alp_dx import AlpDiagnosis   # your ALP wrapper


# -------------------------
# Data structures
# -------------------------

@dataclass
class EDU:
    """
    One discourse unit from the model response.

    Attributes:
      edu_id: unique id (example_id + index).
      text: raw EDU text.
      weight: discourse weight w_i (from RST: nucleus vs satellite).
      ig: information gain IG(c_i, S) (can be precomputed or filled later).
      symptoms: list of Prolog atoms representing observations for ALP.
      claim_atom: Prolog atom representing the central claim for this EDU.
      label: gold label: 1 = hallucination, 0 = non-hallucination, None = unlabeled.
      meta: any extra info (dataset, question, etc.).
    """
    edu_id: str
    text: str
    weight: float
    ig: float
    symptoms: List[str]
    claim_atom: str
    label: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class EDUDecision:
    edu: EDU
    hallucination: bool
    reason: str
    explanation: Optional[List[str]] = None
    score_claim: Optional[float] = None
    score_base: Optional[float] = None


# -------------------------
# Information Gain interface
# -------------------------

class IGComputer:
    """
    Pluggable IG computation.

    You should override compute_ig() to call your auditor from
    halluc_in_health (e.g., NLI/LLM-based IG between source and EDU).
    """

    def compute_ig(self, edu: EDU, source_text: str) -> float:
        """
        Default stub: if edu.ig already set, just return it.
        Replace this with a call to your real IG implementation,
        e.g. via your 'entropy-based hallucination' module.
        """
        return edu.ig


class MedicalIGComputer(IGComputer):
    """
    Example adapter that calls your real entropy / IG code
    from halluc_in_health.

    This is a template — fill in the call to your module.
    """

    def __init__(self, auditor: Any):
        self.auditor = auditor   # e.g. a class instance wrapping your NLI / LLM

    def compute_ig(self, edu: EDU, source_text: str) -> float:
        # Pseudocode – replace with actual call:
        # return self.auditor.compute_kl_divergence(source_text, edu.text)
        return super().compute_ig(edu, source_text)


# -------------------------
# Medical hallucination detector
# -------------------------

class IGAbductionHallucinationDetector:
    """
    Information-Gain + MDL-style score + Abduction + Counter-Abduction
    specialized for medical hallucinations with AlpDiagnosis (gout, RA, etc.)
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

        In your paper, this corresponds to:
          Score(H) ≈ L(H) + Σ w_i L(EDU_i | H)
        approximated for the focal EDU by:
          alpha * |H| + beta * w_i * IG(c_i,S)
        """
        self.alp = alp if alp is not None else AlpDiagnosis()
        self.ig_computer = ig_computer if ig_computer is not None else IGComputer()
        self.ig_low = ig_low_threshold
        self.ig_high = ig_high_threshold
        self.margin = counter_margin
        self.alpha_h = alpha_h
        self.beta_residual = beta_residual

    # ----------------------------------
    # IG update
    # ----------------------------------
    def update_ig(self, edu: EDU, source_text: str) -> None:
        edu.ig = self.ig_computer.compute_ig(edu, source_text)

    # ----------------------------------
    # Abduction via ALP (gout vs RA etc.)
    # ----------------------------------
    def abductive_explanation(self, edu: EDU) -> List[str]:
        """
        Uses your ALP engine to find abductive explanations for the EDU’s symptoms.

        Example:
          symptoms = ["severe_joint_pain", "swelling", "uric_acid_high"]
          explain_obs(symptoms) -> ['disease(gout)']
        """
        if not edu.symptoms:
            return []

        try:
            explanation = self.alp.explain_obs(edu.symptoms)
            # explanation is typically a list of Prolog atoms as strings.
            return explanation
        except Exception:
            return []

    # ----------------------------------
    # MDL-style scoring (hook for halluc_in_health)
    # ----------------------------------
    def score_hypothesis(self, edu: EDU, explanation: List[str]) -> float:
        """
        MDL-style proxy:
          Score(H) = alpha * L(H) + beta * w_i * IG(c_i,S)

        Where:
          - L(H) ~ length / complexity of abduced hypothesis set.
          - residual term uses discourse-weighted IG for this EDU.

        To align with halluc_in_health, you can:
          - Replace this with a call to your logical ensemble/MDL scorer:
              score = my_logical_ensemble_score(edu, explanation)
          - Or incorporate probabilities / defeat scores from Prolog.
        """
        L_H = self.alpha_h * float(len(explanation))
        L_residual = self.beta_residual * (edu.weight * edu.ig)
        return L_H + L_residual

    def baseline_score(self, edu: EDU) -> float:
        """
        Baseline score for explaining context without endorsing the claim.

        Here: minimal residual IG at IG_low, scaled by discourse weight.
        In halluc_in_health, you could instead:
          - use your ensemble's score for a "no-claim" or neutral baseline.
        """
        return self.beta_residual * (edu.weight * self.ig_low)

    # ----------------------------------
    # Single-EDU classification
    # ----------------------------------
    def classify_edu(self, edu: EDU, source_text: str) -> EDUDecision:
        """
        1) Compute/refresh IG.
        2) If IG < low threshold → non-hallucination.
        3) Else try abduction.
        4) Compare ScoreClaim vs ScoreBase (counter-abduction).
        """
        # 1. IG refresh
        self.update_ig(edu, source_text)

        # 2. Low IG → close to source distribution → non-hallucination
        if edu.ig < self.ig_low:
            return EDUDecision(
                edu=edu,
                hallucination=False,
                reason=f"Low IG ({edu.ig:.3f}) — close to source-supported distribution.",
            )

        # 3. Abductive search
        explanation = self.abductive_explanation(edu)

        # If no abductive explanation at all → strong counter-abductive failure
        if not explanation:
            return EDUDecision(
                edu=edu,
                hallucination=True,
                reason=(
                    f"IG={edu.ig:.3f} and no abductive explanation found "
                    f"(counter-abductive failure)."
                ),
                explanation=None,
            )

        # 4. Score with MDL-style objective
        score_claim = self.score_hypothesis(edu, explanation)
        score_base = self.baseline_score(edu)

        if score_claim > score_base + self.margin:
            # Every hypothesis that supports c_i is too costly → hallucination
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

        # Otherwise abduction succeeds → non-hallucination
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
    # Batch processing
    # ----------------------------------
    def analyze_example(self, source_text: str, edus: List[EDU]) -> List[EDUDecision]:
        return [self.classify_edu(e, source_text) for e in edus]
