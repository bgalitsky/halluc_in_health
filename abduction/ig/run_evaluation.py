# run_evaluation.py
from __future__ import annotations
import argparse
from typing import List
from alp_dx import AlpDiagnosis
from abduction.ig.hallucination_detector_alp import IGAbductionHallucinationDetector, EDUDecision
from abduction.ig.data_io import load_dataset


def evaluate_predictions(decisions: List[EDUDecision]):
    """
    Compute basic metrics: accuracy, precision, recall, F1 for hallucination detection.
    Assumes edu.label ∈ {0,1} or None.
    """
    y_true = []
    y_pred = []

    for d in decisions:
        if d.edu.label is None:
            continue
        y_true.append(int(d.edu.label))
        y_pred.append(1 if d.hallucination else 0)

    if not y_true:
        print("No labeled EDUs found in dataset.")
        return

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}, N={len(y_true)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="Path to JSON dataset.")
    ap.add_argument("--ig-low", type=float, default=0.5)
    ap.add_argument("--ig-high", type=float, default=1.5)
    ap.add_argument("--counter-margin", type=float, default=0.1)
    args = ap.parse_args()

    # Load dataset
    examples = load_dataset(args.dataset)

    # Initialize ALP engine and load demo rules
    alp = AlpDiagnosis()
    alp.load_demo()

    detector = IGAbductionHallucinationDetector(
        alp=alp,
        ig_low_threshold=args.ig_low,
        ig_high_threshold=args.ig_high,
        counter_margin=args.counter_margin,
    )

    all_decisions: List[EDUDecision] = []

    for ex in examples:
        source = ex["source"]
        edus = ex["edus"]
        decisions = detector.analyze_example(source_text=source, edus=edus)
        all_decisions.extend(decisions)

        # Optional: print per-example details
        print(f"Example ID: {ex['id']}")
        for d in decisions:
            print(f"  EDU: {d.edu.edu_id}")
            print(f"    Text: {d.edu.text}")
            print(f"    IG: {d.edu.ig:.3f}")
            print(f"    Hallucination: {d.hallucination}")
            print(f"    Reason: {d.reason}")
            print(f"    Explanation: {d.explanation}")
        print("-" * 70)

    # Evaluate against gold labels
    evaluate_predictions(all_decisions)


if __name__ == "__main__":
    main()
