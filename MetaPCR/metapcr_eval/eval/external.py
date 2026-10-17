"""External benchmarks with their ORIGINAL annotations (Section 5.18).

ProcessBench  official F1 = harmonic mean of accuracy on erroneous samples
              (first-error index exact match) and on correct samples (predict -1).
RAGTruth      response-level F1, hallucinated response = positive class.
ProofWriter   three-way query accuracy, open-world (True/False/Unknown kept apart).

Sampling is frozen in a manifest BEFORE inference: release, split, n, seed,
selected source ids, sha256 of the id list. Python's `random` is Mersenne Twister,
matching the paper's stated generator (seed 42081).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Optional, Protocol

from .schema import read_jsonl


# ---------------------------------------------------------------- manifests

def make_manifest(benchmark: str, release: str, split: str, ids: list[str], n: int,
                  seed: int = 42081, cluster_of: Optional[dict] = None) -> dict:
    """Sample n ids (or n whole clusters when cluster_of maps id -> source) without
    looking at any predictions. Unsampled = never scored, never tuned on."""
    rng = random.Random(seed)
    if cluster_of:
        sources = sorted({cluster_of[i] for i in ids})
        chosen_src = set(rng.sample(sources, min(n, len(sources))))
        chosen = sorted(i for i in ids if cluster_of[i] in chosen_src)
    else:
        chosen = sorted(rng.sample(sorted(ids), min(n, len(ids))))
    digest = hashlib.sha256("\n".join(chosen).encode()).hexdigest()
    return {"benchmark": benchmark, "release": release, "split": split, "seed": seed,
            "rng": "MersenneTwister(python random)", "n": len(chosen), "ids": chosen, "sha256": digest}


def save_manifest(m: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(m, indent=1))


def load_manifest(path: str | Path) -> dict:
    m = json.loads(Path(path).read_text())
    if hashlib.sha256("\n".join(m["ids"]).encode()).hexdigest() != m["sha256"]:
        raise ValueError(f"manifest {path} was modified after freezing")
    return m


# ---------------------------------------------------------------- loaders

def load_processbench(path: str | Path) -> list[dict]:
    """Official ProcessBench rows: {id, problem, steps: [...], label: int (-1 = correct)}."""
    rows = read_jsonl(path) if str(path).endswith(".jsonl") else json.loads(Path(path).read_text())
    return [{"instance_id": r["id"], "source_id": r["id"], "problem": r["problem"],
             "steps": r["steps"], "gold_first_error": int(r["label"])} for r in rows]


def load_ragtruth(response_path: str | Path, source_path: str | Path) -> list[dict]:
    """RAGTruth response.jsonl + source_info.jsonl. A response is hallucinated iff it
    has at least one annotated span. source_id groups responses to one source."""
    src = {r["source_id"]: r for r in read_jsonl(source_path)}
    out = []
    for r in read_jsonl(response_path):
        s = src[r["source_id"]]
        out.append({"instance_id": str(r["id"]), "source_id": str(r["source_id"]),
                    "split": r.get("split"), "context": s.get("source_info"), "prompt": s.get("prompt"),
                    "response": r["response"], "gold_halluc": len(r.get("labels") or []) > 0})
    return out


def load_proofwriter(path: str | Path, max_depth: Optional[int] = 5) -> list[dict]:
    """ProofWriter OWA meta-*.jsonl: one theory, many questions. Each question is an
    instance; the theory is the cluster (source_id). Labels kept as True/False/Unknown."""
    out = []
    for th in read_jsonl(path):
        for qid, q in th["questions"].items():
            d = q.get("QDep")
            if max_depth is not None and d is not None and int(d) > max_depth:
                continue
            ans = q["answer"]
            label = {True: "True", False: "False"}.get(ans, str(ans))
            out.append({"instance_id": f'{th["id"]}::{qid}', "source_id": th["id"],
                        "theory": th["theory"], "question": q["question"], "depth": d,
                        "gold_label": label, "gold_proofs": q.get("proofsWithIntermediates") or q.get("proofs")})
    return out


# ---------------------------------------------------------------- verifier hook

class Verifier(Protocol):
    """What a system must implement to be scored on the external tasks.
    Primary inference never receives gold fields (routing, proofs, labels)."""
    name: str
    def first_error(self, problem: str, steps: list[str]) -> int: ...
    def is_hallucinated(self, context: str, prompt: str, response: str) -> Optional[bool]: ...
    def query_label(self, theory: str, question: str) -> str: ...


GOLD_FIELDS = {"gold_first_error", "gold_halluc", "gold_label", "gold_proofs"}


def strip_gold(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in GOLD_FIELDS}


def run_verifier(task: str, verifier: Verifier, rows: Iterable[dict], fallback=None) -> list[dict]:
    """Run a verifier on manifest rows. Exceptions/timeouts are logged and mapped by
    a deterministic fallback frozen before testing — never dropped (Section 5.18)."""
    out = []
    for r in rows:
        x = strip_gold(r)
        rec = {"instance_id": r["instance_id"], "source_id": r["source_id"], "system": verifier.name,
               "status": "ok"}
        try:
            if task == "processbench":
                rec["pred_first_error"] = verifier.first_error(x["problem"], x["steps"])
            elif task == "ragtruth":
                rec["pred_halluc"] = verifier.is_hallucinated(x["context"], x["prompt"], x["response"])
            elif task == "proofwriter":
                rec["pred_label"] = verifier.query_label(x["theory"], x["question"])
        except Exception as e:  # parse failure, timeout, unsupported input
            rec["status"] = f"error:{type(e).__name__}"
            fb = fallback[task] if fallback else {"processbench": -1, "ragtruth": False, "proofwriter": "Unknown"}[task]
            key = {"processbench": "pred_first_error", "ragtruth": "pred_halluc", "proofwriter": "pred_label"}[task]
            rec[key] = fb
        for g in GOLD_FIELDS & r.keys():
            rec[g] = r[g]
        out.append(rec)
    return out


# ---------------------------------------------------------------- scorers

def processbench_f1(rows: list[dict]) -> dict:
    err = [r for r in rows if r["gold_first_error"] != -1]
    ok = [r for r in rows if r["gold_first_error"] == -1]
    a_err = sum(r["pred_first_error"] == r["gold_first_error"] for r in err) / len(err) if err else 0.0
    a_ok = sum(r["pred_first_error"] == -1 for r in ok) / len(ok) if ok else 0.0
    f = 2 * a_err * a_ok / (a_err + a_ok) if (a_err + a_ok) else 0.0
    return {"acc_erroneous": a_err, "acc_correct": a_ok, "f1": f, "n": len(rows)}


def ragtruth_response_f1(rows: list[dict]) -> dict:
    tp = sum(bool(r["pred_halluc"]) and r["gold_halluc"] for r in rows)
    fp = sum(bool(r["pred_halluc"]) and not r["gold_halluc"] for r in rows)
    fn = sum((not r["pred_halluc"]) and r["gold_halluc"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0, "n": len(rows)}


def proofwriter_accuracy(rows: list[dict]) -> dict:
    labels = ("True", "False", "Unknown")
    per = {l: [r for r in rows if r["gold_label"] == l] for l in labels}
    return {"accuracy": sum(r["pred_label"] == r["gold_label"] for r in rows) / len(rows),
            **{f"acc_{l}": (sum(r["pred_label"] == l for r in per[l]) / len(per[l]) if per[l] else None) for l in labels},
            "n": len(rows)}


SCORERS = {"processbench": (processbench_f1, "f1", "processbench_f1"),
           "ragtruth": (ragtruth_response_f1, "f1", "f1"),
           "proofwriter": (proofwriter_accuracy, "accuracy", "query_accuracy")}
