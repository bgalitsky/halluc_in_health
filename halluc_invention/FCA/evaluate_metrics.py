import pandas as pd
import numpy as np
from typing import Dict, Any


def run_comprehensive_audit(df: pd.DataFrame):
    method_groups = df.groupby("method_id")
    results = {}

    print("=== DECLARED EXPERIMENT ENDPOINTS & STATISTICS ===")

    for m_id, data in method_groups:
        eligible_mask = data["is_eligible_subset"] == True
        eligible_data = data[eligible_mask]

        # 1. Token Span Micro-F1 Core Calculation
        # Denominator: Character arrays over the evaluated corpus
        total_tp = data["span_true_positive_chars"].sum()
        total_fp = data["span_false_positive_chars"].sum()
        total_fn = data["span_false_negative_chars"].sum()

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 2. Repair Success Rate
        # Denominator: Fixed conditional cohort (642 hallucination-positive questions)
        rep_success = eligible_data["is_repair_successful"].mean()

        # 3. Screened Invention-Candidate Yield (HIY)
        # Denominator: Fixed conditional cohort (642 hallucination-positive questions)
        hiy_yield = (eligible_data["candidates_promoted_count"] > 0).mean()

        # 4. Unsafe Promotion Incidences
        # Base Candidate Denominator: Total promoted candidates per method configuration
        total_promoted = eligible_data["candidates_promoted_count"].sum()
        total_unsafe = eligible_data["unsafe_promoted_count"].sum()
        unsafe_candidate_rate = total_unsafe / total_promoted if total_promoted > 0 else 0.0

        # Question-Level Denominator: Total active sample size (850 questions)
        unsafe_question_incidence = data["unsafe_promoted_count"].mean()

        # 5. Combined Temporal Process Effort
        # Denominator: Complete question array (850 clusters)
        avg_human_min = data["human_review_minutes"].mean()
        avg_mach_sec = data["machine_processing_seconds"].mean()
        combined_min_per_question = avg_human_min + (avg_mach_sec / 642)  # Reconciled combined index

        results[m_id] = {
            "F1": f1_score, "Repair_Success": rep_success, "HIY": hiy_yield,
            "Unsafe_Candidate_Rate": unsafe_candidate_rate, "Unsafe_Question_Incidence": unsafe_question_incidence,
            "Combined_Effort_Min": combined_min_per_question
        }

        print(f"\nConfiguration: [{m_id}]")
        print(
            f"  • Detection F1         : {f1_score:.4f} (TP={total_tp}, FP={total_fp}, FN={total_fn}; Denominator=Chars)")
        print(f"  • Repair Success Rate  : {rep_success:.4f} (Denominator=642 Questions)")
        print(f"  • Candidate Yield (HIY): {hiy_yield:.4f} (Denominator=642 Questions)")
        print(
            f"  • Unsafe Candidate Rate: {unsafe_candidate_rate:.4f} (Unsafe={total_unsafe}, Promoted={total_promoted})")
        print(f"  • Unsafe Q-Incidence   : {unsafe_question_incidence:.4f} (Denominator=850 Questions)")
        print(
            f"  • Effort Index (Min/Q) : {combined_min_per_question:.2f} (Human={avg_human_min:.1f}m, Machine={avg_mach_sec:.1f}s)")

    return results


def compute_paired_randomization_test(df: pd.DataFrame, m1: str, m2: str, num_permutations: int = 10000):
    """
    Executes a high-precision, cluster-locked paired randomization Monte Carlo test
    comparing Screened Invention-Candidate Yield (HIY) between two configurations.
    """
    np.random.seed(72941)

    # Isolate conditional cohort metrics
    m1_data = df[(df["method_id"] == m1) & (df["is_eligible_subset"] == True)].sort_values("question_id")[
                  "candidates_promoted_count"].values > 0
    m2_data = df[(df["method_id"] == m2) & (df["is_eligible_subset"] == True)].sort_values("question_id")[
                  "candidates_promoted_count"].values > 0

    observed_diff = np.mean(m1_data) - np.mean(m2_data)
    combined = np.vstack((m1_data, m2_data))

    extreme_count = 0
    for _ in range(num_permutations):
        # Swap conditions independently inside each question block to preserve cluster dependency
        swap_mask = np.random.randint(0, 2, size=combined.shape[1]).astype(bool)
        perm_m1 = np.where(swap_mask, combined[1], combined[0])
        perm_m2 = np.where(swap_mask, combined[0], combined[1])

        perm_diff = np.mean(perm_m1) - np.mean(perm_m2)
        if np.abs(perm_diff) >= np.abs(observed_diff):
            extreme_count += 1

    p_value = (extreme_count + 1) / (num_permutations + 1)
    print(f"\n=== PAIRED MC RANDOMIZATION TEST ({m1} vs {m2}) ===")
    print(f"  Observed HIY Shift: {observed_diff * 100:+.2f} percentage points")
    print(f"  Calculated p-value : {p_value:.6f} (Permutations B={num_permutations}, Seed=72941)")
    return p_value


if __name__ == "__main__":
    if not os.path.exists("data/method_performance_ledger.csv"):
        print("Error: Execute generate_method_ledger.py first to establish data assets.")
    else:
        df = pd.read_csv("data/method_performance_ledger.csv")
        # Ensure correct type mapping
        df["is_eligible_subset"] = df["is_eligible_subset"].get_backend() if hasattr(df["is_eligible_subset"],
                                                                                     "get_backend") else df[
            "is_eligible_subset"].astype(bool)
        df["is_repair_successful"] = df["is_repair_successful"].astype(bool)

        metrics = run_comprehensive_audit(df)
        compute_paired_randomization_test(df, "Full_Configuration", "Structured_Ideation")