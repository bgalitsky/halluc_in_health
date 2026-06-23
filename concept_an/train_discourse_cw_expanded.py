#!/usr/bin/env python3
"""
train_discourse_cw.py

Train a small Concept-Whitening-style classifier on the synthetic
discourse hallucination dataset.

Input:
  discourse_cw_hallucination_dataset_1000.csv

Output:
  discourse_cw_model.pt
  discourse_cw_metrics.json

Usage:
  python train_discourse_cw.py --data discourse_cw_hallucination_dataset_1000.csv
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Candidate feature columns. Only those present in the loaded CSV are used, so
# the same script works on the original 8-feature dataset and on the richer
# complex-trees datasets. ID columns (id, source_id) are deliberately excluded.
FEATURE_COLUMNS = [
    # original 8 discourse features
    "unsupported_nucleus",
    "evidence_density",
    "contrast_density",
    "closure_strength",
    "weak_to_strong_inference",
    "support_balance",
    "defeater_presence",
    "nucleus_pressure",
    # tree-structure features (present in the complex-trees datasets)
    "tree_nodes",
    "tree_edges",
    "tree_depth",
    "nucleus_count",
    "satellite_count",
    "contrast_count",
    "defeater_count",
    "unsupported_count",
    "qualification_count",
    "tree_instability",
    "qualification_level",
    "tree_complexity",
    # immune-extension dataset features
    "immune_ambiguity",
    "bait_score",
]

HALLUCINATION_CONCEPTS = {
    "unsupported_nucleus_promotion",
    "contradiction_omission",
    "defeater_suppression",
    "premature_closure",
    "weak_evidence_amplification",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ConceptWhiteningLayer(nn.Module):
    """
    Practical Concept-Whitening-style layer.

    Full CW learns whitening and rotation jointly. This compact version uses
    BatchNorm as a whitening approximation and learns an orthogonal rotation Q.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, affine=False)
        self.Q_raw = nn.Parameter(torch.eye(dim))

    def orthogonal_Q(self) -> torch.Tensor:
        Q, R = torch.linalg.qr(self.Q_raw)
        sign = torch.sign(torch.diag(R))
        sign[sign == 0] = 1
        return Q * sign.unsqueeze(0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_white = self.bn(z)
        return z_white @ self.orthogonal_Q()


class DiscourseCWClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_concepts: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
            nn.ReLU(),
        )
        self.cw = ConceptWhiteningLayer(num_concepts)
        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_cw = self.cw(z)
        logits = self.classifier(z_cw)
        return logits, z_cw


def concept_alignment_loss(z_cw: torch.Tensor, concept_ids: torch.Tensor) -> torch.Tensor:
    """
    Force discourse_subtype j to activate CW axis j.
    This is the CAP-style alignment objective.
    """
    return F.cross_entropy(z_cw, concept_ids)


def hallucination_axis_score(z_cw: torch.Tensor, concept_names: List[str],
                             hallucination_concepts=None) -> torch.Tensor:
    if hallucination_concepts is None:
        hallucination_concepts = HALLUCINATION_CONCEPTS
    probs = torch.softmax(z_cw, dim=1)
    h_axes = [i for i, name in enumerate(concept_names) if name in hallucination_concepts]
    g_axes = [i for i, name in enumerate(concept_names) if name not in hallucination_concepts]
    if not h_axes:
        # no concept axis maps to a hallucination concept -> score is undefined
        return torch.zeros(z_cw.shape[0], device=z_cw.device)
    h = probs[:, h_axes].sum(dim=1)
    g = probs[:, g_axes].sum(dim=1)
    return h / (h + g + 1e-8)


def load_dataset(path: str):
    df = pd.read_csv(path)
    missing = [c for c in ["label", "discourse_subtype"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use whichever candidate feature columns are present in this CSV.
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not feature_cols:
        raise ValueError("None of FEATURE_COLUMNS are present in the dataset.")

    # Concepts whose samples are labelled hallucinated (derived from the data,
    # so the hallucination-axis read-out works regardless of concept naming).
    subtype_to_label = {
        str(st): str(lab)
        for st, lab in df.groupby("discourse_subtype")["label"].agg(lambda s: s.astype(str).iloc[0]).items()
    }
    hallucination_concepts = {st for st, lab in subtype_to_label.items() if "halluc" in lab.lower()}

    label_enc = LabelEncoder()
    concept_enc = LabelEncoder()

    y = label_enc.fit_transform(df["label"].astype(str))
    c = concept_enc.fit_transform(df["discourse_subtype"].astype(str))
    X = df[feature_cols].astype(float).values

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y[train_idx], dtype=torch.long),
        "c_train": torch.tensor(c[train_idx], dtype=torch.long),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y[test_idx], dtype=torch.long),
        "c_test": torch.tensor(c[test_idx], dtype=torch.long),
        "test_df": df.iloc[test_idx].reset_index(drop=True),
        "label_names": list(label_enc.classes_),
        "concept_names": list(concept_enc.classes_),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "hallucination_concepts": hallucination_concepts,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def train(args):
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    data = load_dataset(args.data)

    model = DiscourseCWClassifier(
        input_dim=data["n_features"],
        hidden_dim=args.hidden_dim,
        num_classes=len(data["label_names"]),
        num_concepts=len(data["concept_names"]),
    ).to(device)

    X_train = data["X_train"].to(device)
    y_train = data["y_train"].to(device)
    c_train = data["c_train"].to(device)
    X_test = data["X_test"].to(device)
    y_test = data["y_test"].to(device)
    c_test = data["c_test"].to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = X_train.shape[0]

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0

        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            logits, z_cw = model(X_train[idx])

            cls_loss = F.cross_entropy(logits, y_train[idx])
            con_loss = concept_alignment_loss(z_cw, c_train[idx])

            Q = model.cw.Q_raw
            I = torch.eye(Q.shape[0], device=device)
            ortho_loss = ((Q.T @ Q - I) ** 2).mean()

            loss = cls_loss + args.concept_weight * con_loss + args.orthogonal_weight * ortho_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)

        if epoch == 1 or epoch % args.print_every == 0:
            model.eval()
            with torch.no_grad():
                logits, z_cw = model(X_test)
                y_pred = logits.argmax(dim=1)
                c_pred = z_cw.argmax(dim=1)
                acc = (y_pred == y_test).float().mean().item()
                cap = (c_pred == c_test).float().mean().item()
            print(f"epoch={epoch:04d} loss={epoch_loss/n:.4f} test_acc={acc:.3f} CAP={cap:.3f}")

    model.eval()
    with torch.no_grad():
        logits, z_cw = model(X_test)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        c_pred = z_cw.argmax(dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()
        c_true = c_test.cpu().numpy()
        h_score = hallucination_axis_score(z_cw, data["concept_names"], data["hallucination_concepts"]).cpu().numpy()

    metrics = {
        "classification_accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "concept_axis_accuracy_CAP": float(accuracy_score(c_true, c_pred)),
        "label_names": data["label_names"],
        "concept_names": data["concept_names"],
        "feature_columns": data["feature_columns"],
        "classification_report": classification_report(
            y_true, y_pred, target_names=data["label_names"], output_dict=True
        ),
        "scaler_mean": data["scaler_mean"],
        "scaler_scale": data["scaler_scale"],
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": data["feature_columns"],
            "label_names": data["label_names"],
            "concept_names": data["concept_names"],
            "scaler_mean": data["scaler_mean"],
            "scaler_scale": data["scaler_scale"],
            "hidden_dim": args.hidden_dim,
        },
        args.out_model,
    )

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model:", args.out_model)
    print("Saved metrics:", args.out_metrics)
    print(json.dumps({
        "classification_accuracy": metrics["classification_accuracy"],
        "classification_macro_f1": metrics["classification_macro_f1"],
        "concept_axis_accuracy_CAP": metrics["concept_axis_accuracy_CAP"],
    }, indent=2))

    test_df = data["test_df"].copy()
    test_df["pred_label"] = [data["label_names"][i] for i in y_pred]
    test_df["pred_concept_axis"] = [data["concept_names"][i] for i in c_pred]
    test_df["cw_hallucination_axis_score"] = np.round(h_score, 3)

    text_col = next((c for c in ["text", "reasoning_text", "patient_text"] if c in test_df.columns), None)
    cols = [
        "id", "label", "pred_label", "discourse_subtype",
        "pred_concept_axis", "cw_hallucination_axis_score",
    ]
    if text_col:
        cols.append(text_col)
    print("\nExample predictions:")
    print(test_df[cols].head(10).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="discourse_cw_hallucination_dataset_1000.csv")
    p.add_argument("--out_model", default="discourse_cw_model.pt")
    p.add_argument("--out_metrics", default="discourse_cw_metrics.json")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--concept_weight", type=float, default=0.5)
    p.add_argument("--orthogonal_weight", type=float, default=0.01)
    p.add_argument("--print_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
