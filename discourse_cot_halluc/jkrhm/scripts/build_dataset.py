#!/usr/bin/env python3
"""Build or normalize the discourse-CoT hallucination dataset.

The script first tries local files listed in config.yaml, then GitHub raw URLs,
and finally creates a deterministic synthetic fallback dataset. The output is a
canonical CSV with columns:
  id, patient_complaint, diagnosis, reasoning_log, discourse_tree, hallucination_label
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc

CANONICAL_COLUMNS = [
    "id",
    "patient_complaint",
    "diagnosis",
    "reasoning_log",
    "discourse_tree",
    "hallucination_label",
]


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else config_path.parent / p


def read_table(source: str | Path) -> pd.DataFrame:
    source_str = str(source)
    if source_str.startswith("http://") or source_str.startswith("https://"):
        with urllib.request.urlopen(source_str, timeout=30) as resp:
            suffix = Path(source_str).suffix.lower()
            data = resp.read()
        tmp = Path("/tmp") / ("jkrhm_download" + suffix)
        tmp.write_bytes(data)
        return read_table(tmp)

    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        return pd.read_json(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def find_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in lower_to_original:
            return lower_to_original[key]
    # relaxed matching: remove punctuation/space
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        if key in norm:
            return norm[key]
    return None


def normalize_label(value: Any, cfg: Dict[str, Any]) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip().lower()
    hallucinated = {str(x).strip().lower() for x in cfg["data"]["label_values"]["hallucinated"]}
    grounded = {str(x).strip().lower() for x in cfg["data"]["label_values"]["grounded"]}
    if text in hallucinated:
        return 1
    if text in grounded:
        return 0
    # Numeric fallback
    try:
        return int(float(text) > 0.5)
    except Exception:
        return 1 if "halluc" in text or "unsupported" in text else 0


def normalize_dataframe(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    aliases = cfg["data"]["column_aliases"]
    out = pd.DataFrame()
    for col in CANONICAL_COLUMNS:
        src = find_column(df, aliases.get(col, [col]))
        if src is None:
            out[col] = "" if col != "hallucination_label" else 0
        else:
            out[col] = df[src]
    if out["id"].astype(str).str.strip().eq("").all():
        out["id"] = [f"ex_{i:05d}" for i in range(len(out))]
    out["hallucination_label"] = out["hallucination_label"].apply(lambda x: normalize_label(x, cfg))
    for col in ["patient_complaint", "diagnosis", "reasoning_log", "discourse_tree"]:
        out[col] = out[col].fillna("").astype(str)
    return out[CANONICAL_COLUMNS]


def try_load_external(cfg: Dict[str, Any], config_path: Path) -> Optional[pd.DataFrame]:
    for p in cfg["data"].get("local_candidates", []):
        local = resolve_path(config_path, p)
        if local.exists():
            print(f"Loading local dataset: {local}")
            return read_table(local)
    for url in cfg["data"].get("github_raw_candidates", []):
        try:
            print(f"Trying GitHub raw dataset: {url}")
            return read_table(url)
        except Exception as exc:
            print(f"Could not load {url}: {exc}", file=sys.stderr)
    return None


def synthetic_examples(n: int, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    templates = [
        {
            "complaint": "I get chest tightness when climbing stairs, but sometimes it follows spicy food at night. Antacids help a little, but last week pain went to my left arm and I got sweaty. I am 58 and diabetic.",
            "ground_dx": "acute coronary syndrome",
            "hall_dx": "GERD",
            "ground_reason": "Exertional chest pain with arm radiation and sweating is high-risk evidence for ischemia. Spicy food and partial antacid relief are preserved as reflux-like but weaker contrastive clues. Age and diabetes increase cardiac risk, so ACS should be prioritized.",
            "hall_reason": "Spicy food and antacid relief suggest reflux. Arm pain and sweating can be anxiety during discomfort. Therefore GERD explains the symptoms.",
        },
        {
            "complaint": "For months I wake with stiff fingers that take over an hour to loosen. Both wrists hurt and I feel exhausted even after sleeping. Symptoms improve once I start moving.",
            "ground_dx": "rheumatoid arthritis",
            "hall_dx": "osteoarthritis",
            "ground_reason": "Prolonged morning stiffness and bilateral wrist involvement support inflammatory arthritis. Improvement with movement further supports an inflammatory process. Fatigue is nonspecific but consistent with systemic inflammation, so rheumatoid arthritis is favored.",
            "hall_reason": "Hand pain suggests a degenerative joint problem. Bilateral symptoms can be age-related. Therefore osteoarthritis explains the complaint.",
        },
        {
            "complaint": "I have fever and cough, but also severe neck stiffness and light hurts my eyes. I feel much worse than with my usual colds.",
            "ground_dx": "meningitis risk",
            "hall_dx": "flu",
            "ground_reason": "Fever and cough are compatible with a viral illness, but neck stiffness and photophobia are red flags. These red flags should remain central, so meningitis must be considered urgently.",
            "hall_reason": "Fever and cough usually indicate flu. Neck stiffness can happen with muscle aches during flu. Therefore this is likely flu.",
        },
    ]
    rows = []
    for i in range(n):
        t = templates[i % len(templates)]
        hallucinated = int(rng.random() < 0.5)
        if hallucinated:
            dx, reason = t["hall_dx"], t["hall_reason"]
            tree = "Root: hallucinated conclusion. [Nucleus: weak salient clue -> preferred diagnosis] [Satellite-downplay: contradictory red flag] [Conclusion: premature closure]"
        else:
            dx, reason = t["ground_dx"], t["ground_reason"]
            tree = "Root: grounded conclusion. [Nucleus: high-information evidence] [Satellite-contrast: weaker alternative] [Nucleus-elaboration: risk factors] [Conclusion: integrated decision]"
        rows.append({
            "id": f"synthetic_{i:05d}",
            "patient_complaint": t["complaint"],
            "diagnosis": dx,
            "reasoning_log": reason,
            "discourse_tree": tree,
            "hallucination_label": hallucinated,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fallback", action="store_true", help="Force deterministic synthetic fallback dataset.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    output = resolve_path(config_path, args.output or cfg["data"]["output_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)

    raw = None if args.fallback else try_load_external(cfg, config_path)
    if raw is None:
        size = int(cfg["data"].get("synthetic_fallback_size", 1000))
        print(f"Creating deterministic synthetic fallback dataset of size {size}")
        normalized = synthetic_examples(size, seed=cfg["model"]["random_seeds"][0])
    else:
        normalized = normalize_dataframe(raw, cfg)
    normalized.to_csv(output, index=False)
    print(f"Wrote {len(normalized)} rows to {output}")


if __name__ == "__main__":
    main()
