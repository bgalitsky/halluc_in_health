import pandas as pd
import numpy as np
import os


def verify_partition_balance(ledger_path: str, seed: int = 4242):
    if not os.path.exists(ledger_path):
        print(f"Error: Target registry data missing at {ledger_path}")
        return

    df = pd.read_csv(ledger_path)

    # Isolate to question level to verify covariate initialization properties
    q_df = df.drop_duplicates(subset=["question_id"]).copy()

    # Assign stratified domain blocks based on token mapping structures
    # matching the three master sectors (CHEM, SYS, MECH)
    q_df["sector"] = q_df["question_id"].apply(
        lambda idx: "CHEM" if int(idx.split("-")[-1]) <= 270 else
        ("SYS" if int(idx.split("-")[-1]) <= 580 else "MECH")
    )

    np.random.seed(seed)
    unique_qs = q_df["question_id"].values
    np.random.shuffle(unique_qs)

    # Generate strict 20/80 calibration index boundary partitions
    split_idx = int(len(unique_qs) * 0.20)
    calibration_qs = set(unique_qs[:split_idx])
    estimation_qs = set(unique_qs[split_idx:])

    q_df["partition"] = q_df["question_id"].apply(
        lambda x: "Calibration (20%)" if x in calibration_qs else "Estimation (80%)"
    )

    print("=== COVARIATE BALANCING SPLIT VERIFICATION ===")
    print(f"Total Unique Query Clusters Audited: {len(q_df)}")
    print(f"  • Calibration Partition Pool      : {len(calibration_qs)} questions")
    print(f"  • Estimation Partition Pool       : {len(estimation_qs)} questions\n")

    # Evaluate distribution matching across sectors
    balance_table = pd.crosstab(q_df["sector"], q_df["partition"], normalize="columns") * 100
    print("Sector Stratification Distribution Balance (Percentage Values):")
    print(balance_table.round(2).to_string())

    # Assert maximum acceptable sampling divergence threshold (5%)
    for sector in balance_table.index:
        diff = abs(balance_table.loc[sector, "Calibration (20%)"] - balance_table.loc[sector, "Estimation (80%)"])
        assert diff < 5.0, f"🔴 Warning: Asymmetric distribution detected in {sector} split: {diff:.2f}% gap."

    print("\n🟢 CROSS-VALIDATION MATRIX STATUS: BALANCED (No variance anomalies found).")


if __name__ == "__main__":
    verify_partition_balance("data/method_performance_ledger.csv")