#!/usr/bin/env python3
"""Construct simple discourse/reasoning trees from CoT traces.

This is a deterministic heuristic parser intended as a reproducible fallback for
paper experiments. It can be replaced by an RST parser or LLM parser while keeping
the same JSONL tree schema.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else config_path.parent / p


def split_edus(text: str) -> List[str]:
    bracket_parts = re.findall(r"\[([^\[\]]+)\]", text or "")
    if bracket_parts:
        return [p.strip() for p in bracket_parts if p.strip()]
    parts = re.split(r"(?<=[.!?;])\s+|\s+\-\s+|\s+\|\s+", text or "")
    return [p.strip() for p in parts if p.strip()]


def infer_relation(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["contrast", "however", "but", "although", "whereas", "alternative"]):
        return "contrast"
    if any(x in t for x in ["concession", "despite", "even though"]):
        return "concession"
    if any(x in t for x in ["because", "therefore", "so", "thus", "->", "⇒", "due to"]):
        return "cause"
    if any(x in t for x in ["evidence", "support", "supports", "indicate", "suggest"]):
        return "evidence"
    if any(x in t for x in ["downplay", "ignore", "ignored", "dismiss", "anxiety", "age-related"]):
        return "downplay"
    if any(x in t for x in ["red flag", "urgent", "risk", "high-risk", "danger"]):
        return "red_flag"
    if any(x in t for x in ["conclusion", "prioritize", "favored", "most likely", "explains"]):
        return "conclusion"
    return "elaboration"


def infer_nuclearity(text: str, relation: str, position: int) -> str:
    t = text.lower()
    if "nucleus" in t or relation in {"conclusion", "red_flag"}:
        return "nucleus"
    if "satellite" in t or relation in {"contrast", "concession", "downplay"}:
        return "satellite"
    return "nucleus" if position == 0 else "satellite"


def infer_role(text: str, relation: str) -> str:
    t = text.lower()
    if relation == "conclusion" or "conclusion" in t:
        return "conclusion"
    if relation in {"contrast", "concession"}:
        return "counter_evidence"
    if relation == "downplay":
        return "downplay" if "downplay" in t else "ignored_evidence"
    if relation == "red_flag":
        return "counter_evidence"
    if relation in {"evidence", "cause"}:
        return "evidence"
    if any(x in t for x in ["claim", "favor", "diagnosis"]):
        return "claim"
    return "background"


def parse_trace(example_id: str, trace: str, given_tree: str = "") -> Dict[str, Any]:
    source = given_tree if str(given_tree).strip() else trace
    edus = split_edus(source)
    if not edus:
        edus = [str(trace or "").strip() or "empty reasoning trace"]
    nodes = []
    for i, edu in enumerate(edus):
        rel = infer_relation(edu)
        nuc = infer_nuclearity(edu, rel, i)
        nodes.append({
            "id": i,
            "text": edu,
            "relation": rel,
            "nuclearity": nuc,
            "role": infer_role(edu, rel),
        })
    # Simple rooted tree: first nucleus/conclusion-like node is root, all other nodes attach to it.
    root = 0
    for node in nodes:
        if node["nuclearity"] == "nucleus" and node["relation"] in {"conclusion", "evidence", "red_flag"}:
            root = node["id"]
            break
    edges = []
    for node in nodes:
        if node["id"] != root:
            edges.append({"parent": root, "child": node["id"], "relation": node["relation"]})
    return {"id": example_id, "root": root, "nodes": nodes, "edges": edges}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    input_path = resolve_path(config_path, args.input or cfg["data"]["output_csv"])
    output_path = resolve_path(config_path, args.output or cfg["outputs"]["trees_jsonl"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            tree = parse_trace(str(row.get("id", "")), str(row.get("reasoning_log", "")), str(row.get("discourse_tree", "")))
            f.write(json.dumps(tree, ensure_ascii=False) + "\n")
    print(f"Wrote {len(df)} discourse trees to {output_path}")


if __name__ == "__main__":
    main()
