import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


"""
How it works

The classifier computes features such as:

whether the tree contains Satellite-ignore

whether it contains Satellite-downplay

whether contradictory evidence is preserved via contrast or background

whether the Nucleus seems to rely on weak clues

whether likely red-flag evidence is being ignored

Then it applies a transparent decision tree:

Ignored red flags → hallucination

Weak nucleus + no preserved contradiction → hallucination

Several dismissive operations → hallucination

Contrast/background present and score low → not hallucination

otherwise use the total score

Expected behavior on your tree style

This should classify trees like:

Valid

Root: favor ACS. [Nucleus: exertional pain + radiation + sweating -> ischemia]
[Satellite-contrast: spicy food/nighttime + antacid relief -> GERD-like]
[Nucleus-elaboration: age + diabetes increase cardiac risk]
[Conclusion: prioritize ACS]

as not hallucination

and trees like:

Hallucinated

Root: favor GERD. [Nucleus: spicy food + antacid relief -> reflux]
[Satellite-downplay: arm radiation + sweating -> anxiety response]
[Satellite-ignore: exertional trigger + diabetes]
[Conclusion: GERD explains symptoms]

as hallucination

"""

@dataclass
class TreeFeatures:
    has_ignore: bool
    has_downplay: bool
    has_dismissal: bool
    has_overclaim: bool
    has_reinterpretation: bool
    has_generalization: bool
    has_contrast: bool
    has_background: bool
    has_elaboration: bool
    has_support: bool
    has_satellite_ignore_or_dismiss: bool
    root_favors_low_specificity: bool
    conclusion_overconfident: bool
    contradiction_preserved: bool
    red_flag_ignored: bool
    score_hallucination: int


class HallucinationDecisionTree:
    """
    Rule-based decision tree for discourse-tree classification.

    Input format expected:
      Root: favor GERD. [Nucleus: meals/spicy food + antacid relief -> reflux]
      [Satellite-downplay: radiation + sweating -> anxiety response]
      [Satellite-ignore: exertional trigger + cardiac risk]
      [Conclusion: GERD explains symptoms]

    or similar compact discourse-tree strings.
    """

    LOW_SPECIFICITY_HINTS = {
        "meals", "spicy food", "wine", "stress", "anxiety", "panic",
        "family history", "overweight", "body aches", "dehydration",
        "subjective", "indigestion", "lifestyle", "cramping", "soreness"
    }

    HIGH_SPECIFICITY_HINTS = {
        "radiation", "diaphoresis", "exertional", "unilateral calf swelling",
        "worsening dyspnea", "morning stiffness", "symmetric small-joint",
        "rlq", "movement pain", "weight loss", "tremor", "eye changes",
        "vomiting", "flank", "fever", "polyuria", "polydipsia",
        "blurry vision", "ketosis", "ascending from feet", "balance in dark"
    }

    OVERCONFIDENT_CONCLUSION_HINTS = {
        "confirms", "must be", "best explanation", "simple diagnosis",
        "straightforward", "does not matter", "incidental", "only"
    }

    RED_FLAG_HINTS = {
        "radiation", "sweating", "diaphoresis", "cardiac risk", "exertional",
        "unilateral calf", "swelling", "worsening dyspnea", "eye changes",
        "weight loss", "vomiting", "fever", "flank", "ketosis", "rlq"
    }

    def extract_segments(self, discourse_tree: str) -> List[Tuple[str, str]]:
        """
        Extract bracketed segments like:
            [Nucleus: ...]
            [Satellite-downplay: ...]
            [Conclusion: ...]
        Returns list of (label, text).
        """
        segments = re.findall(r"\[([^\]:]+):\s*([^\]]+)\]", discourse_tree)
        return [(label.strip().lower(), text.strip().lower()) for label, text in segments]

    def get_root_text(self, discourse_tree: str) -> str:
        m = re.search(r"root:\s*([^.]+)\.", discourse_tree, re.IGNORECASE)
        return m.group(1).strip().lower() if m else ""

    def has_any(self, text: str, vocab: set) -> bool:
        return any(term in text for term in vocab)

    def count_any(self, text: str, vocab: set) -> int:
        return sum(1 for term in vocab if term in text)

    def extract_features(self, discourse_tree: str) -> TreeFeatures:
        raw = discourse_tree.lower()
        root_text = self.get_root_text(raw)
        segments = self.extract_segments(raw)

        labels = [label for label, _ in segments]
        texts = [text for _, text in segments]

        has_ignore = any("ignore" in lbl for lbl in labels)
        has_downplay = any("downplay" in lbl for lbl in labels)
        has_dismissal = any("dismissal" in lbl or "dismiss" in lbl for lbl in labels)
        has_overclaim = any("overclaim" in lbl for lbl in labels)
        has_reinterpretation = any("reinterpretation" in lbl or "reinterpret" in lbl for lbl in labels)
        has_generalization = any("generalization" in lbl or "generalize" in lbl for lbl in labels)

        has_contrast = any("contrast" in lbl for lbl in labels)
        has_background = any("background" in lbl for lbl in labels)
        has_elaboration = any("elaboration" in lbl for lbl in labels)
        has_support = any("support" in lbl for lbl in labels)

        has_satellite_ignore_or_dismiss = any(
            ("satellite" in lbl) and (
                "ignore" in lbl or "dismiss" in lbl or "downplay" in lbl or
                "reinterpret" in lbl or "overclaim" in lbl
            )
            for lbl in labels
        )

        nucleus_texts = [text for label, text in segments if "nucleus" in label]
        satellite_texts = [text for label, text in segments if "satellite" in label]
        conclusion_texts = [text for label, text in segments if "conclusion" in label]

        nucleus_joined = " ".join(nucleus_texts)
        satellite_joined = " ".join(satellite_texts)
        conclusion_joined = " ".join(conclusion_texts)

        low_specificity_nucleus_count = self.count_any(nucleus_joined, self.LOW_SPECIFICITY_HINTS)
        high_specificity_nucleus_count = self.count_any(nucleus_joined, self.HIGH_SPECIFICITY_HINTS)
        root_favors_low_specificity = low_specificity_nucleus_count > 0 and high_specificity_nucleus_count == 0

        contradiction_preserved = has_contrast or has_background

        conclusion_overconfident = self.has_any(conclusion_joined, self.OVERCONFIDENT_CONCLUSION_HINTS)

        # Red flags appear in satellites, especially with ignore/downplay/dismissal labels
        red_flag_ignored = (
            any(self.has_any(text, self.RED_FLAG_HINTS) for text in satellite_texts)
            and has_satellite_ignore_or_dismiss
        )

        score_hallucination = 0
        if has_ignore:
            score_hallucination += 3
        if has_downplay:
            score_hallucination += 2
        if has_dismissal:
            score_hallucination += 2
        if has_overclaim:
            score_hallucination += 2
        if has_reinterpretation:
            score_hallucination += 2
        if has_generalization:
            score_hallucination += 1
        if has_satellite_ignore_or_dismiss:
            score_hallucination += 2
        if root_favors_low_specificity:
            score_hallucination += 2
        if conclusion_overconfident:
            score_hallucination += 1
        if red_flag_ignored:
            score_hallucination += 3

        # Protective / grounded features reduce score
        if has_contrast:
            score_hallucination -= 2
        if has_background:
            score_hallucination -= 1
        if has_elaboration:
            score_hallucination -= 1
        if has_support:
            score_hallucination -= 1
        if contradiction_preserved:
            score_hallucination -= 1

        return TreeFeatures(
            has_ignore=has_ignore,
            has_downplay=has_downplay,
            has_dismissal=has_dismissal,
            has_overclaim=has_overclaim,
            has_reinterpretation=has_reinterpretation,
            has_generalization=has_generalization,
            has_contrast=has_contrast,
            has_background=has_background,
            has_elaboration=has_elaboration,
            has_support=has_support,
            has_satellite_ignore_or_dismiss=has_satellite_ignore_or_dismiss,
            root_favors_low_specificity=root_favors_low_specificity,
            conclusion_overconfident=conclusion_overconfident,
            contradiction_preserved=contradiction_preserved,
            red_flag_ignored=red_flag_ignored,
            score_hallucination=score_hallucination,
        )

    def classify(self, discourse_tree: str) -> Dict:
        """
        Explicit decision tree:
        1) If red flags are ignored in dismissive satellites -> hallucination.
        2) Else if root uses low-specificity nucleus AND contradiction not preserved -> hallucination.
        3) Else if multiple dismissive markers exist -> hallucination.
        4) Else if contrast/background/elaboration dominate and hallucination score low -> not hallucination.
        5) Else threshold on total score.
        """
        f = self.extract_features(discourse_tree)

        if f.red_flag_ignored:
            label = "hallucination"
            reason = "Red-flag evidence is placed into dismissive/ignoring satellite branches."
        elif f.root_favors_low_specificity and not f.contradiction_preserved:
            label = "hallucination"
            reason = "Weak clues anchor the nucleus while contradiction is not preserved."
        elif (f.has_ignore + f.has_downplay + f.has_dismissal + f.has_overclaim + f.has_reinterpretation) >= 2:
            label = "hallucination"
            reason = "Multiple dismissive or unsupported discourse operations are present."
        elif (f.has_contrast or f.has_background) and f.score_hallucination <= 1:
            label = "not hallucination"
            reason = "Contradiction is preserved and the overall discourse structure is integrative."
        elif f.score_hallucination >= 3:
            label = "hallucination"
            reason = f"Aggregate hallucination score is high ({f.score_hallucination})."
        else:
            label = "not hallucination"
            reason = f"Aggregate hallucination score is low ({f.score_hallucination}) and the tree remains comparatively balanced."

        return {
            "label": label,
            "reason": reason,
            "features": asdict(f),
        }


if __name__ == "__main__":
    clf = HallucinationDecisionTree()

    examples = [
        "Root: favor ACS. [Nucleus: exertional pain + radiation + sweating -> ischemia] "
        "[Satellite-contrast: meal/spicy-food burning + antacid relief -> GERD-like] "
        "[Nucleus-elaboration: age/risk factors increase cardiac risk] "
        "[Conclusion: prioritize ACS]",

        "Root: favor GERD. [Nucleus: meals/spicy food + antacid relief -> reflux] "
        "[Satellite-downplay: radiation + sweating -> anxiety response] "
        "[Satellite-ignore: exertional trigger + cardiac risk] "
        "[Conclusion: GERD explains symptoms]",

        "Root: favor pyelonephritis. [Satellite-background: dysuria + frequency -> cystitis] "
        "[Nucleus: fever + flank/back pain + vomiting + systemic illness] "
        "[Satellite-contrast: burning improved] [Conclusion: pyelonephritis]",

        "Root: favor cystitis. [Nucleus: dysuria + frequency] "
        "[Satellite-reinterpretation: back pain + vomiting -> dehydration/strain] "
        "[Satellite-overclaim: burning improved -> infection resolving] "
        "[Conclusion: simple cystitis]",
    ]

    for i, ex in enumerate(examples, 1):
        result = clf.classify(ex)
        print("=" * 80)
        print(f"Example {i}")
        print(ex)
        print("Prediction:", result["label"])
        print("Reason:", result["reason"])
        print("Features:")
        for k, v in result["features"].items():
            print(f"  {k}: {v}")