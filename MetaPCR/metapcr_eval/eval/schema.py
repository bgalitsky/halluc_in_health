"""Record formats and loaders for the MetaPCR-LLM evaluation (Section 5).

Every analysis reads JSON Lines files. One line = one (system, instance) prediction.
Fields not needed by a given analysis may be omitted.

InstanceRecord (predictions/*.jsonl)
-----------------------------------
instance_id        str   unique per evaluated instance
source_id          str   original source problem/document; variants share it (cluster unit)
dataset            str   Truthful-Halluc | Med-Halluc | eSNLI-Halluc | Autoimmune-narrate-halluc | ...
regime             str   rule | uncertainty | abduction | conflict | constraint | mixed
system             str   e.g. "Full MetaPCR-LLM", "ValidLLP-style", "ablation:-native_certificates"
panel / origin     str   optional: composition-control panel and case origin (released|transformed|stress|local_pass)
gold_halluc        bool  gold hallucination label (positive class)
pred_halluc        bool|null  predicted label; null = indeterminate (no decisive verdict)
p_halluc           float predicted probability of hallucination (calibration layer frozen on dev)
gold_escalate      bool  gold "should go to a human" decision
pred_escalate      bool
accepted           bool  output accepted by the validator (coverage numerator)
violates_hard_constraint bool  accepted output violates a gold hard constraint
steps              list  [{step_id, gold_valid: bool, pred_valid: bool|null}] on the SHARED gold decomposition
chain_gold_valid   bool
chain_pred_valid   bool|null  (native certificate graph, or induced decision for baselines)
bridges            list|null  [{bridge_id, gold_valid, pred_valid}] ; null = system has no bridge objects
gold_defect_step   str|null   gold first defective step (null if chain valid)
pred_defect_step   str|null
gold_failure_type  str|null ; pred_failure_type str|null
gold_validator     str|null ; pred_validator str|null
gold_bridge_failure str|null ; pred_bridge_failure str|null
latency            dict  {decomp, native, bridges, counter_abd, aggregation, wall} in seconds
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:  # fail loudly: silent drops bias results
                raise ValueError(f"{path}:{ln}: invalid JSON ({e})") from e
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def by_system(rows: Iterable[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["system"]].append(r)
    return dict(out)


def group(rows: Iterable[dict], key: str) -> Iterator[tuple[str, list[dict]]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[r.get(key)].append(r)
    for k in sorted(g, key=str):
        yield k, g[k]


def check_paired(a: list[dict], b: list[dict], name_a: str, name_b: str) -> None:
    """Paired comparisons require identical instance sets (Section 5.5: same test partition)."""
    ia = {r["instance_id"] for r in a}
    ib = {r["instance_id"] for r in b}
    if ia != ib:
        raise ValueError(
            f"{name_a} vs {name_b}: instance sets differ "
            f"(only-{name_a}={len(ia - ib)}, only-{name_b}={len(ib - ia)}). "
            "Unsuccessful instances must be scored, not dropped (Section 5.18)."
        )
