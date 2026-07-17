#!/usr/bin/env python3
"""Extract lexical, discourse-tree, domain, and JKRHM-style proxy features."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else config_path.parent / p


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+", str(text).lower())


def count_terms(text: str, terms: Iterable[str]) -> int:
    t = str(text).lower()
    return sum(t.count(str(term).lower()) for term in terms)


def entropy(counts: Iterable[int]) -> float:
    vals = np.array([c for c in counts if c > 0], dtype=float)
    if vals.size == 0:
        return 0.0
    p = vals / vals.sum()
    return float(-(p * np.log(p)).sum())


def tree_depth(tree: Dict[str, Any]) -> int:
    edges = tree.get("edges", [])
    if not edges:
        return 1
    children = {}
    for e in edges:
        children.setdefault(int(e["parent"]), []).append(int(e["child"]))
    root = int(tree.get("root", 0))
    q = deque([(root, 1)])
    max_depth = 1
    seen = set()
    while q:
        node, depth = q.popleft()
        if node in seen:
            continue
        seen.add(node)
        max_depth = max(max_depth, depth)
        for child in children.get(node, []):
            q.append((child, depth + 1))
    return max_depth


def lexical_features(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, float]:
    text = " ".join([str(row.get("patient_complaint", "")), str(row.get("diagnosis", "")), str(row.get("reasoning_log", ""))])
    toks = tokenize(text)
    sents = [s for s in re.split(r"[.!?]+", str(row.get("reasoning_log", ""))) if s.strip()]
    f = {
        "lex_num_tokens": float(len(toks)),
        "lex_num_unique": float(len(set(toks))),
        "lex_num_sentences": float(len(sents)),
        "lex_avg_sentence_len": float(len(toks) / max(len(sents), 1)),
        "lex_uncertainty_count": float(count_terms(text, cfg["features"]["lexical_style"]["uncertainty_terms"])),
        "lex_overconfidence_count": float(count_terms(text, cfg["features"]["lexical_style"]["overconfidence_terms"])),
        "lex_unsupported_attr_count": float(count_terms(text, cfg["features"]["lexical_style"]["unsupported_attribution_terms"])),
    }
    return f


def discourse_features(tree: Dict[str, Any], row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, float]:
    nodes = tree.get("nodes", [])
    edges = tree.get("edges", [])
    rels = [n.get("relation", "") for n in nodes]
    nucs = [n.get("nuclearity", "") for n in nodes]
    roles = [n.get("role", "") for n in nodes]
    rel_counter = Counter(rels)
    node_count = len(nodes)
    edge_count = len(edges)
    nucleus_count = sum(1 for x in nucs if x == "nucleus")
    satellite_count = sum(1 for x in nucs if x == "satellite")
    depth = tree_depth(tree)
    f: Dict[str, float] = {
        "top_node_count": float(node_count),
        "top_edge_count": float(edge_count),
        "top_depth": float(depth),
        "top_branching": float(edge_count / max(node_count, 1)),
        "top_leaf_proxy": float(max(node_count - edge_count, 1)),
        "disc_relation_entropy": entropy(rel_counter.values()),
        "ns_nucleus_count": float(nucleus_count),
        "ns_satellite_count": float(satellite_count),
        "ns_ratio": float(nucleus_count / max(satellite_count, 1)),
        "disc_contrast_count": float(rel_counter.get("contrast", 0)),
        "disc_concession_count": float(rel_counter.get("concession", 0)),
        "disc_evidence_count": float(rel_counter.get("evidence", 0) + rel_counter.get("cause", 0)),
        "disc_elaboration_count": float(rel_counter.get("elaboration", 0)),
        "disc_downplay_count": float(rel_counter.get("downplay", 0)),
        "disc_red_flag_count": float(rel_counter.get("red_flag", 0)),
        "role_counter_evidence_count": float(sum(1 for r in roles if r == "counter_evidence")),
        "role_ignored_evidence_count": float(sum(1 for r in roles if r == "ignored_evidence")),
        "role_conclusion_count": float(sum(1 for r in roles if r == "conclusion")),
    }
    # Simple derived signals.
    f["disc_contrast_fraction"] = f["disc_contrast_count"] / max(node_count, 1)
    f["disc_evidence_fraction"] = f["disc_evidence_count"] / max(node_count, 1)
    f["disc_premature_closure"] = 1.0 if nodes and nodes[0].get("relation") == "conclusion" else 0.0
    root = int(tree.get("root", 0)) if nodes else 0
    root_text = nodes[root].get("text", "").lower() if nodes and root < len(nodes) else ""
    weak_terms = ["generic", "weak", "spicy", "antacid", "hand pain", "usually", "age-related"]
    f["disc_weak_nucleus_flag"] = float(any(t in root_text for t in weak_terms))
    return f


def domain_features(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, float]:
    text = " ".join([str(row.get("patient_complaint", "")), str(row.get("diagnosis", "")), str(row.get("reasoning_log", ""))]).lower()
    medical_terms = cfg["features"]["domain_cues"]["medical_terms"]
    return {
        "domain_medical_term_count": float(count_terms(text, medical_terms)),
        "domain_redflag_word_count": float(count_terms(text, ["urgent", "red flag", "severe", "diabetic", "sweaty", "photophobia", "neck stiffness"])),
    }


def add_jkrhm_proxy(f: Dict[str, float], cfg: Dict[str, Any]) -> None:
    # Operational proxies for the JKRHM components described in the paper.
    det_proxy = math.log1p(f.get("lex_num_unique", 0.0) + f.get("disc_relation_entropy", 0.0) + f.get("disc_evidence_count", 0.0))
    sigma_proxy = math.log1p(f.get("disc_premature_closure", 0.0) + f.get("lex_overconfidence_count", 0.0) + f.get("role_ignored_evidence_count", 0.0))
    kappa_proxy = math.log1p(abs(f.get("ns_ratio", 1.0) - 1.0) + 1.0 / (f.get("disc_contrast_count", 0.0) + 1.0))
    dcounter = max(0.0, 1.0 - f.get("disc_contrast_fraction", 0.0)) + f.get("role_ignored_evidence_count", 0.0) + f.get("disc_downplay_count", 0.0)
    disc_score = (
        0.25 * min(f.get("disc_evidence_fraction", 0.0), 1.0)
        + 0.25 * min(f.get("disc_contrast_fraction", 0.0) * 5.0, 1.0)
        + 0.20 * min(f.get("disc_relation_entropy", 0.0) / 2.0, 1.0)
        + 0.15 * min(f.get("top_depth", 0.0) / 5.0, 1.0)
        + 0.15 * (1.0 - min(f.get("disc_weak_nucleus_flag", 0.0), 1.0))
    )
    lam = float(cfg["features"]["jkrhm"].get("lambda_counter", 0.5))
    mu = float(cfg["features"]["jkrhm"].get("mu_discourse", 0.5))
    risk = -det_proxy + sigma_proxy + 2.0 * kappa_proxy + lam * dcounter + mu * (1.0 - disc_score)
    f.update({
        "jkrhm_log_det_k_proxy": det_proxy,
        "jkrhm_log_sigma_max_proxy": sigma_proxy,
        "jkrhm_log_kappa_proxy": kappa_proxy,
        "jkrhm_dcounter_proxy": dcounter,
        "jkrhm_disc_score": disc_score,
        "jkrhm_risk": risk,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--trees", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    dataset_path = resolve_path(config_path, args.dataset or cfg["data"]["output_csv"])
    trees_path = resolve_path(config_path, args.trees or cfg["outputs"]["trees_jsonl"])
    output_path = resolve_path(config_path, args.output or cfg["outputs"]["features_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    trees = {}
    with open(trees_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            trees[str(item["id"])] = item

    rows = []
    for _, row in df.iterrows():
        fdict: Dict[str, float] = {}
        fdict.update(lexical_features(row, cfg))
        fdict.update(discourse_features(trees.get(str(row["id"]), {"nodes": [], "edges": []}), row, cfg))
        fdict.update(domain_features(row, cfg))
        add_jkrhm_proxy(fdict, cfg)
        fdict["id"] = row["id"]
        fdict["hallucination_label"] = int(row["hallucination_label"])
        rows.append(fdict)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote feature matrix with {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
