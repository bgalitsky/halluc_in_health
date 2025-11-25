# data_io.py
from __future__ import annotations
import json
from typing import List, Dict, Any
from hallucination_detector_alp import EDU


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load dataset JSON.

    Returns a list of examples:
      {
        "id": str,
        "source": str,
        "edus": List[EDU]
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples_out = []
    for ex in data.get("examples", []):
        ex_id = ex.get("id", "")
        source = ex.get("source", "")
        edus_json = ex.get("edus", [])
        edus_objs: List[EDU] = []
        for e in edus_json:
            edu = EDU(
                edu_id=e["edu_id"],
                text=e["text"],
                weight=float(e.get("weight", 1.0)),
                ig=float(e.get("ig")) if e.get("ig") is not None else 0.0,
                symptoms=e.get("symptoms", []),
                claim_atom=e.get("claim_atom", ""),
                label=e.get("label", None),
            )
            edus_objs.append(edu)

        examples_out.append({
            "id": ex_id,
            "source": source,
            "edus": edus_objs
        })

    return examples_out
