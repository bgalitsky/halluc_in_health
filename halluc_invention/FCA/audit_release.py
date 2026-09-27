import os
import csv
import sys
from typing import Dict, Any


class EvaluationDataAuditor:
    def __init__(self, ledger_csv_path: str):
        self.ledger_path = ledger_csv_path

    def audit_dataset_integrity(self) -> Dict[str, Any]:
        if not os.path.exists(self.ledger_path):
            print(f"Error: Target data trace not found at {self.ledger_path}", file=sys.stderr)
            return {}

        total_rows = 0
        unique_questions = set()
        screened_count = 0
        promoted_count = 0
        status_distribution = {}

        with open(self.ledger_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                unique_questions.add(row["question_id"])

                if row["is_screened"].lower() == "true":
                    screened_count += 1
                if row["is_promoted"].lower() == "true":
                    promoted_count += 1

                label = row["status_label"]
                status_distribution[label] = status_distribution.get(label, 0) + 1

        return {
            "total_traces": total_rows,
            "distinct_questions": len(unique_questions),
            "screened_candidates": screened_count,
            "promoted_candidates": promoted_count,
            "distribution": status_distribution
        }

    def verify_manuscript_compliance(self, metrics: Dict[str, Any]) -> bool:
        print("=== COMMENCING REPRODUCTION GATE AUDIT ===")
        print(f"Total Evaluated Traces: {metrics['total_traces']} (Manuscript Expected: 1845)")
        print(f"Distinct Question Blocks: {metrics['distinct_questions']} (Manuscript Expected: 642)")
        print(f"Screened Gate Entry: {metrics['screened_candidates']} (Manuscript Expected: 912)")
        print(f"Promoted Candidates: {metrics['promoted_candidates']} (Manuscript Expected: 514)")
        print("-" * 42)

        # Strict validation checks matching reported metrics
        assertions = [
            metrics['total_traces'] == 1845,
            metrics['distinct_questions'] == 642,
            metrics['screened_candidates'] == 912,
            metrics['promoted_candidates'] == 514,
            metrics['distribution'].get('stable_unique_candidate', 0) == 514,
            metrics['distribution'].get('prior_art_collision', 0) == 398
        ]

        passed = all(assertions)
        if passed:
            print("🟢 REPRODUCTION AUDIT STATUS: COMPLIANT WITH MANUSCRIPT SPECIFICATION")
        else:
            print("🔴 REPRODUCTION AUDIT STATUS: NON-COMPLIANT (DISCREPANCY DETECTED)")
        return passed


if __name__ == "__main__":
    # Ensure standard directory pathing exists for integration testing
    os.makedirs("data", exist_ok=True)
    auditor = EvaluationDataAuditor("data/ledger_v1.csv")
    metrics = auditor.audit_dataset_integrity()
    if metrics:
        auditor.verify_manuscript_compliance(metrics)