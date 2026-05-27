# discourse_cw_analyzer.py
# Prototype: Discourse-aware Concept Whitening style reasoning analyzer

import re
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple


@dataclass
class EDU:
    idx: int
    text: str
    role: str          # nucleus / satellite
    relation: str      # Evidence, Contrast, Cause, Elaboration, Conclusion, Unknown


@dataclass
class ConceptScore:
    concept: str
    activation: float
    explanation: str


class DiscourseCWAnalyzer:
    """
    Lightweight prototype for analyzing reasoning text using discourse concepts.

    It approximates:
    1. EDU segmentation
    2. RST-like relation detection
    3. discourse concept extraction
    4. Concept Whitening-style axis activation
    5. hallucination risk scoring
    """

    def __init__(self):
        self.concepts = [
            "unsupported_nucleus_promotion",
            "contradiction_omission",
            "defeater_suppression",
            "premature_closure",
            "weak_evidence_amplification",
            "stable_evidence_integration",
            "contrast_preservation",
            "qualified_conclusion",
        ]

        # Concept axes: hand-coded prototype weights over features.
        # In a trained CW model, these would be learned orthogonal axes.
        self.concept_axes = {
            "unsupported_nucleus_promotion": np.array([1.2, -0.5, -0.8, 1.0, 0.9, 0.2, -0.7, 0.6]),
            "contradiction_omission":        np.array([0.8, -0.2, -1.2, 0.9, 1.0, 0.4, -1.1, 0.5]),
            "defeater_suppression":         np.array([0.7, -0.3, -1.3, 0.8, 0.7, 0.2, -1.2, 0.6]),
            "premature_closure":            np.array([0.9, -0.6, -0.4, 1.3, 1.1, 0.1, -0.8, 0.9]),
            "weak_evidence_amplification":  np.array([1.1, -0.7, -0.5, 0.9, 1.2, 0.3, -0.6, 0.6]),
            "stable_evidence_integration":  np.array([-0.8, 1.2, 0.7, -0.5, -0.6, 1.2, 0.7, -0.4]),
            "contrast_preservation":        np.array([-0.7, 0.9, 1.4, -0.4, -0.5, 0.8, 1.3, -0.5]),
            "qualified_conclusion":         np.array([-0.6, 0.8, 0.7, -0.7, -0.8, 0.9, 0.6, -1.1]),
        }

    def analyze(self, text: str) -> Dict:
        edus = self.segment_edus(text)
        discourse_tree = self.assign_discourse_roles(edus)
        features = self.extract_features(discourse_tree)
        whitened = self.whiten(features)
        concept_scores = self.compute_concept_scores(whitened)
        hallucination_score = self.compute_hallucination_score(concept_scores)
        verdict = self.verdict(hallucination_score)

        return {
            "verdict": verdict,
            "hallucination_score": round(hallucination_score, 3),
            "features": self.feature_names_with_values(features),
            "discourse_units": [asdict(e) for e in discourse_tree],
            "concept_scores": [asdict(c) for c in concept_scores],
            "summary": self.generate_summary(verdict, hallucination_score, concept_scores),
        }

    def segment_edus(self, text: str) -> List[EDU]:
        parts = re.split(r"(?<=[.!?])\s+|;\s+|\n+", text.strip())
        parts = [p.strip() for p in parts if p.strip()]
        return [EDU(i, p, "unknown", "Unknown") for i, p in enumerate(parts)]

    def assign_discourse_roles(self, edus: List[EDU]) -> List[EDU]:
        for edu in edus:
            t = edu.text.lower()

            if any(x in t for x in ["because", "therefore", "thus", "so", "hence", "as a result"]):
                edu.relation = "Cause/Result"
                edu.role = "nucleus"

            elif any(x in t for x in ["however", "although", "but", "nevertheless", "on the other hand"]):
                edu.relation = "Contrast"
                edu.role = "nucleus"

            elif any(x in t for x in ["for example", "for instance", "such as", "evidence", "shows", "indicates"]):
                edu.relation = "Evidence"
                edu.role = "satellite"

            elif any(x in t for x in ["maybe", "possibly", "might", "could", "suggests", "appears"]):
                edu.relation = "Weak Evidence"
                edu.role = "satellite"

            elif any(x in t for x in ["must", "definitely", "clearly", "proves", "certainly", "there is no doubt"]):
                edu.relation = "Strong Conclusion"
                edu.role = "nucleus"

            elif any(x in t for x in ["unless", "except", "alternative", "another explanation", "differential"]):
                edu.relation = "Defeater/Alternative"
                edu.role = "satellite"

            else:
                edu.relation = "Elaboration"
                edu.role = "satellite"

        return edus

    def extract_features(self, edus: List[EDU]) -> np.ndarray:
        n = max(len(edus), 1)

        weak = sum(e.relation == "Weak Evidence" for e in edus)
        evidence = sum(e.relation == "Evidence" for e in edus)
        contrast = sum(e.relation == "Contrast" for e in edus)
        strong_conclusion = sum(e.relation == "Strong Conclusion" for e in edus)
        cause_result = sum(e.relation == "Cause/Result" for e in edus)
        defeater = sum(e.relation == "Defeater/Alternative" for e in edus)
        nucleus = sum(e.role == "nucleus" for e in edus)
        satellite = sum(e.role == "satellite" for e in edus)

        unsupported_nucleus = max(0, strong_conclusion - evidence - contrast - defeater)
        evidence_density = evidence / n
        contrast_density = contrast / n
        closure_strength = strong_conclusion / n
        weak_to_strong = weak * strong_conclusion / n
        support_balance = evidence / (strong_conclusion + 1)
        defeater_presence = defeater / n
        nucleus_pressure = nucleus / (satellite + 1)

        return np.array([
            unsupported_nucleus,
            evidence_density,
            contrast_density,
            closure_strength,
            weak_to_strong,
            support_balance,
            defeater_presence,
            nucleus_pressure,
        ], dtype=float)

    def whiten(self, x: np.ndarray) -> np.ndarray:
        """
        Prototype whitening for one sample.
        In real CW training, whitening uses dataset covariance.
        Here we normalize feature vector to approximate decorrelation/scaling.
        """
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        return (x - mean) / std

    def compute_concept_scores(self, z: np.ndarray) -> List[ConceptScore]:
        scores = []

        for concept, axis in self.concept_axes.items():
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            activation = float(np.dot(z, axis))
            activation = 1 / (1 + math.exp(-activation))  # sigmoid

            scores.append(
                ConceptScore(
                    concept=concept,
                    activation=round(activation, 3),
                    explanation=self.explain_concept(concept, activation),
                )
            )

        scores.sort(key=lambda s: s.activation, reverse=True)
        return scores

    def compute_hallucination_score(self, scores: List[ConceptScore]) -> float:
        hallucination_concepts = {
            "unsupported_nucleus_promotion",
            "contradiction_omission",
            "defeater_suppression",
            "premature_closure",
            "weak_evidence_amplification",
        }

        grounded_concepts = {
            "stable_evidence_integration",
            "contrast_preservation",
            "qualified_conclusion",
        }

        h = sum(s.activation for s in scores if s.concept in hallucination_concepts)
        g = sum(s.activation for s in scores if s.concept in grounded_concepts)

        return h / (h + g + 1e-8)

    def verdict(self, score: float) -> str:
        if score >= 0.68:
            return "Likely hallucinated / structurally unstable reasoning"
        if score >= 0.52:
            return "Mixed or uncertain reasoning"
        return "Likely grounded / structurally stable reasoning"

    def explain_concept(self, concept: str, activation: float) -> str:
        explanations = {
            "unsupported_nucleus_promotion":
                "A strong central claim appears insufficiently supported by evidence.",
            "contradiction_omission":
                "The reasoning does not preserve enough contrast or counter-evidence.",
            "defeater_suppression":
                "Alternative explanations or defeaters appear underrepresented.",
            "premature_closure":
                "The text reaches a strong conclusion before enough support is established.",
            "weak_evidence_amplification":
                "Weak evidence appears to be amplified into a strong conclusion.",
            "stable_evidence_integration":
                "Evidence is integrated in a structurally stable way.",
            "contrast_preservation":
                "Contrasts and alternatives are explicitly represented.",
            "qualified_conclusion":
                "The conclusion appears qualified rather than overstated.",
        }

        strength = "high" if activation > 0.7 else "moderate" if activation > 0.45 else "low"
        return f"{strength.capitalize()} activation: {explanations[concept]}"

    def feature_names_with_values(self, x: np.ndarray) -> Dict[str, float]:
        names = [
            "unsupported_nucleus",
            "evidence_density",
            "contrast_density",
            "closure_strength",
            "weak_to_strong_inference",
            "support_balance",
            "defeater_presence",
            "nucleus_pressure",
        ]
        return {name: round(float(value), 3) for name, value in zip(names, x)}

    def generate_summary(
        self,
        verdict: str,
        score: float,
        concept_scores: List[ConceptScore]
    ) -> str:
        top = concept_scores[:3]
        concepts = ", ".join(f"{c.concept}={c.activation}" for c in top)
        return (
            f"{verdict}. Hallucination score={score:.3f}. "
            f"Top discourse concept activations: {concepts}."
        )


if __name__ == "__main__":
    sample = """
    The patient has mild abdominal discomfort and fatigue. This clearly proves colon cancer.
    There is no doubt because fatigue is often seen in cancer. Therefore, the diagnosis is certain.
    """

    analyzer = DiscourseCWAnalyzer()
    result = analyzer.analyze(sample)

    print("\n=== VERDICT ===")
    print(result["verdict"])
    print("\n=== SCORE ===")
    print(result["hallucination_score"])
    print("\n=== FEATURES ===")
    for k, v in result["features"].items():
        print(f"{k}: {v}")

    print("\n=== DISCOURSE UNITS ===")
    for edu in result["discourse_units"]:
        print(edu)

    print("\n=== CONCEPT SCORES ===")
    for score in result["concept_scores"]:
        print(score)

    print("\n=== SUMMARY ===")
    print(result["summary"])