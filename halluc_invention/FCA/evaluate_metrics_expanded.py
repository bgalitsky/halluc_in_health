import pandas as pd
import numpy as np
import os
import sys


def calculate_metric_point_estimates(df_sample: pd.DataFrame, method_id: str) -> dict:
    """Computes exact metric definitions over a provided dataframe slice."""
    m_data = df_sample[df_sample["method_id"] == method_id]
    eligible_data = m_data[m_data["is_eligible_subset"] == True]

    # 1. Detection F1 (Denominator: pooled characters)
    total_tp = m_data["span_true_positive_chars"].sum()
    total_fp = m_data["span_false_positive_chars"].sum()
    total_fn = m_data["span_false_negative_chars"].sum()
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # 2. Repair Success Rate (Denominator: fixed conditional questions)
    rep_success = eligible_data["is_repair_successful"].mean() if len(eligible_data) > 0 else 0

    # 3. Screened Invention-Candidate Yield (HIY) (Denominator: fixed conditional questions)
    hiy_yield = (eligible_data["candidates_promoted_count"] > 0).mean() if len(eligible_data) > 0 else 0

    return {"F1": f1, "Repair_Success": rep_success, "HIY": hiy_yield}


def execute_question_cluster_bootstrap(ledger_path: str, n_bootstraps: int = 10000, seed: int = 72941):
    if not os.path.exists(ledger_path):
        print(f"Error: Target data trace not found at {ledger_path}", file=sys.stderr)
        return

    df = pd.read_csv(ledger_path)
    df["is_eligible_subset"] = df["is_eligible_subset"].astype(bool)
    df["is_repair_successful"] = df["is_repair_successful"].astype(bool)

    # Extract unique question clusters to lock structural dependencies
    unique_questions = df["question_id"].unique()
    n_clusters = len(unique_questions)

    methods = df["method_id"].unique()

    # Pre-allocate bootstrap accumulation registers
    boot_records = {m: {"F1": [], "Repair_Success": [], "HIY": []} for m in methods}

    print(f"Executing {n_bootstraps} Question-Cluster Bootstrap draws (Seed: {seed})...")
    np.random.seed(seed)

    # Generate an explicit map from question_id to row indexes for optimization
    q_to_indices = {q: df.index[df["question_id"] == q].tolist() for q in unique_questions}

    for b in range(1, n_bootstraps + 1):
        # Resample question clusters with replacement
        boot_questions = np.random.choice(unique_questions, size=n_clusters, replace=True)

        # Reconstruct the resampled dataframe slice via index pooling
        sampled_indices = []
        for q in boot_questions:
            sampled_indices.extend(q_to_indices[q])

        df_boot_sample = df.iloc[sampled_indices]

        for m in methods:
            m_metrics = calculate_metric_point_estimates(df_boot_sample, m)
            boot_records[m]["F1"].append(m_metrics["F1"])
            boot_records[m]["Repair_Success"].append(m_metrics["Repair_Success"])
            boot_records[m]["HIY"].append(m_metrics["HIY"])

        if b % 2000 == 0 or b == n_bootstraps:
            print(f"  • Progress: Draw {b}/{n_bootstraps} completed.")

    print("\n" + "=" * 60)
    print("MANUSCRIPT TABLE 6 REPRODUCIBLE BOUNDARY SUMMARY (95% Percentile CI)")
    print("=" * 60)

    for m in sorted(methods):
        print(f"\nConfiguration: {m}")
        for metric in ["F1", "Repair_Success", "HIY"]:
            arr = np.sort(np.array(boot_records[m][metric]))
            point_est = calculate_metric_point_estimates(df, m)[metric]

            # Extract lower and upper 95% percentile confidence bounds
            low_bound = np.percentile(arr, 2.5)
            high_bound = np.percentile(arr, 97.5)
            print(f"  • {metric:<14}: {point_est:.2f} [{low_bound:.2f}, {high_bound:.2f}]")


if __name__ == "__main__":
    execute_question_cluster_bootstrap("data/method_performance_ledger.csv", n_bootstraps=10000)