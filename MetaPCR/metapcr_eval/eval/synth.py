"""Synthetic inputs in the exact file formats the pipeline expects.

PURPOSE: exercise every script end to end and give a worked example of each
input file. The numbers it produces are NOT the paper's results and must never
be reported as such. Replace data/ with real system outputs.

System skill levels are loosely shaped after the paper's tables so the output
looks familiar, and errors are correlated across systems through a shared
per-instance difficulty, which is what makes paired tests meaningful.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .schema import write_jsonl

DATASETS = {  # name: (test n, hallucination rate, regime distribution)
    "Truthful-Halluc": (500, 0.46, [0.20, 0.16, 0.07, 0.18, 0.11, 0.28]),
    "Med-Halluc": (1000, 0.53, [0.20, 0.17, 0.08, 0.10, 0.14, 0.31]),
    "eSNLI-Halluc": (500, 0.40, [0.16, 0.20, 0.08, 0.07, 0.15, 0.34]),
    "Autoimmune-narrate-halluc": (600, 0.38, [0.18, 0.12, 0.11, 0.13, 0.19, 0.27]),
}
REGIMES = ["rule", "uncertainty", "abduction", "conflict", "constraint", "mixed"]
VALIDATORS = ["prolog", "z3", "alp", "problog", "argumentation", "temporal", "lean"]
FAILURE_TYPES = ["unsupported_premise", "invalid_inference", "constraint_violation",
                 "defeated_explanation", "cross_logic_contradiction", "translation_failure"]

# error rate (≈ 1 - accuracy), per-regime extra error, step/bridge/loc skill, has_bridges, latency scale
SYSTEMS = {
    "LP": dict(err=0.34, reg={"uncertainty": .06, "conflict": .04}, step=.70, loc=.62, bridges=False, lat=.6),
    "LP + IC": dict(err=0.30, reg={"uncertainty": .05}, step=.74, loc=.64, bridges=False, lat=.65),
    "PLP": dict(err=0.23, reg={"uncertainty": -.10, "conflict": .06}, step=.80, loc=.66, bridges=False, lat=.65),
    "ALP": dict(err=0.24, reg={"abduction": -.10, "conflict": .08}, step=.75, loc=.68, bridges=False, lat=.7),
    "Argumentation/DeLP": dict(err=0.22, reg={"conflict": -.10, "uncertainty": .06}, step=.76, loc=.70, bridges=False, lat=.7),
    "SMT/CLP": dict(err=0.27, reg={"constraint": -.12}, step=.80, loc=.70, bridges=False, lat=.7),
    "Single-logic PCR": dict(err=0.28, reg={"mixed": .04}, step=.82, loc=.72, bridges=False, lat=.74),
    "ValidLLP-style": dict(err=0.17, reg={"mixed": .04, "Med-Halluc": -.03}, step=.84, loc=.78, bridges=False, lat=.80),
    "MetaPCR w/o bridges": dict(err=0.17, reg={"mixed": .08}, step=.84, loc=.74, bridges=False, lat=.82),
    "Full MetaPCR-LLM": dict(err=0.14, reg={"constraint": .04, "mixed": .02}, step=.88, loc=.80, bridges=True, lat=.90),
}
ABLATIONS = {
    "ablation:-provenance_labels": .25, "ablation:-uncertainty_labels": .31, "ablation:-temporal_labels": .35,
    "ablation:-defeasibility_labels": .39, "ablation:-native_certificates": .43,
    "ablation:-bridge_validation": .17, "ablation:-counter_abduction": .26, "ablation:single_target_logic": .36,
}


def _instances(rng):
    out = []
    for d, (n, rate, dist) in DATASETS.items():
        i = 0
        while i < n:
            src = f"{d}:src{len(out)}"
            for _ in range(int(rng.integers(1, 4))):  # 1–3 variants per source share a cluster
                if i >= n:
                    break
                regime = REGIMES[int(rng.choice(6, p=dist))]
                halluc = bool(rng.random() < rate)
                nsteps = int(rng.integers(3, 7))
                defect = int(rng.integers(nsteps)) if halluc else None
                steps = [{"step_id": f"s{k}", "gold_valid": k != defect} for k in range(nsteps)]
                nbr = int(rng.integers(1, 4)) if regime == "mixed" else 0
                bad_bridge = halluc and regime == "mixed" and rng.random() < 0.5
                bridges = [{"bridge_id": f"b{k}", "gold_valid": not (bad_bridge and k == 0)} for k in range(nbr)]
                out.append({"instance_id": f"{d}:{i}", "source_id": src, "dataset": d, "regime": regime,
                            "gold_halluc": halluc, "steps": steps, "bridges": bridges,
                            "gold_defect_step": None if defect is None else f"s{defect}",
                            "gold_failure_type": None if not halluc else FAILURE_TYPES[int(rng.integers(6))],
                            "gold_validator": None if not halluc else VALIDATORS[int(rng.integers(7))],
                            "gold_bridge_failure": "b0" if bad_bridge else None,
                            "gold_conflict": bool(halluc and rng.random() < .4) if regime == "mixed" else None,
                            "constraint": regime in ("constraint", "mixed") and rng.random() < .5,
                            "u": float(rng.random())})
                i += 1
    return out


def _predict(inst, name, prof, rng, err=None):
    e = (err if err is not None else prof["err"]) + prof["reg"].get(inst["regime"], 0) + prof["reg"].get(inst["dataset"], 0)
    e = min(max(e, .02), .6)
    wrong = (0.75 * inst["u"] + 0.25 * rng.random()) < e
    pred = inst["gold_halluc"] != wrong
    steps = [{"step_id": s["step_id"], "gold_valid": s["gold_valid"],
              "pred_valid": s["gold_valid"] if rng.random() < prof["step"] else not s["gold_valid"]} for s in inst["steps"]]
    chain_gold = not inst["gold_halluc"]
    chain_pred = chain_gold if rng.random() < (1 - e + 0.02) else not chain_gold
    bridges = None
    if prof["bridges"]:
        bridges = [{**b, "pred_valid": b["gold_valid"] if rng.random() < .88 else not b["gold_valid"]} for b in inst["bridges"]]
    loc_ok = rng.random() < prof["loc"]
    p = float(np.clip((0.85 if pred else 0.15) + rng.normal(0, .12), 0.01, 0.99))
    accepted = not pred
    gold_esc = inst["u"] > 0.8 or (inst["gold_halluc"] and inst["regime"] == "mixed" and inst["u"] > .5)
    lat_scale = prof["lat"]
    return {
        "instance_id": inst["instance_id"], "source_id": inst["source_id"], "dataset": inst["dataset"],
        "regime": inst["regime"], "system": name, "gold_halluc": inst["gold_halluc"], "pred_halluc": pred,
        "p_halluc": p, "gold_escalate": gold_esc,
        "pred_escalate": gold_esc if rng.random() < (0.78 if prof["bridges"] else 0.70) else not gold_esc,
        "accepted": accepted, "violates_hard_constraint": accepted and inst["gold_halluc"] and inst["constraint"] and rng.random() < .3,
        "steps": steps, "chain_gold_valid": chain_gold, "chain_pred_valid": chain_pred, "bridges": bridges,
        "gold_defect_step": inst["gold_defect_step"],
        "pred_defect_step": inst["gold_defect_step"] if loc_ok else "s_wrong",
        "gold_failure_type": inst["gold_failure_type"],
        "pred_failure_type": inst["gold_failure_type"] if rng.random() < prof["loc"] - .04 else "other",
        "gold_validator": inst["gold_validator"],
        "pred_validator": (inst["gold_validator"] if rng.random() < prof["loc"] - .09 else "other") if name != "LLM self-evaluation" else None,
        "gold_bridge_failure": inst["gold_bridge_failure"],
        "pred_bridge_failure": (inst["gold_bridge_failure"] if rng.random() < .76 else "none") if prof["bridges"] else None,
        "gold_conflict": inst["gold_conflict"],
        "pred_conflict": None if inst["gold_conflict"] is None else (inst["gold_conflict"] if rng.random() < (.85 if prof["bridges"] else .66) else not inst["gold_conflict"]),
        "latency": {"decomp": float(rng.lognormal(np.log(.2), .3)), "native": float(rng.lognormal(np.log(.4 * lat_scale), .3)),
                    "bridges": float(rng.lognormal(np.log(.3), .3)) if prof["bridges"] else None,
                    "counter_abd": float(rng.lognormal(np.log(.24), .3)), "aggregation": float(rng.lognormal(np.log(.18), .3)),
                    "wall": float(rng.lognormal(np.log(lat_scale), .08))},
    }


def generate(out_dir: str | Path, seed: int = 20261016) -> None:
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    inst = _instances(rng)

    primary = [_predict(i, s, p, rng) for s, p in SYSTEMS.items() for i in inst]
    full = SYSTEMS["Full MetaPCR-LLM"]
    for s, e in ABLATIONS.items():
        prof = dict(full, bridges=s != "ablation:-bridge_validation")
        primary += [_predict(i, s, prof, rng, err=e) for i in inst]
    write_jsonl(out / "primary_predictions.jsonl", primary)

    # dataset partitions (train/validation/test) for Tables 2–3
    parts = []
    for i in inst:
        parts.append({"instance_id": i["instance_id"], "source_id": i["source_id"], "dataset": i["dataset"],
                      "partition": "test", "gold_halluc": i["gold_halluc"], "regime": i["regime"]})
    for d, (n, rate, dist) in DATASETS.items():
        for part, frac in (("train", .4), ("validation", .6)):
            for k in range(int(n * frac)):
                parts.append({"instance_id": f"{d}:{part}{k}", "source_id": f"{d}:{part}src{k}", "dataset": d,
                              "partition": part, "gold_halluc": bool(rng.random() < rate),
                              "regime": REGIMES[int(rng.choice(6, p=dist))]})
    write_jsonl(out / "dataset_instances.jsonl", parts)

    # multi-LLM recomposition (Table 12)
    rec = []
    strat = {"Best single model": (.25, .29), "Whole-answer voting": (.23, .27), "Step recomposition": (.20, .23),
             "+ native certificates": (.15, .20), "+ certificates + bridges + labels": (.14, .18)}
    for s, (e, bad) in strat.items():
        for i in inst:
            r = _predict(i, s, dict(full, err=e, bridges=False), rng)
            n_c = int(rng.integers(2, 6))
            r["n_compositions"] = n_c
            r["n_invalid_compositions"] = int(rng.binomial(n_c, bad))
            rec.append(r)
    write_jsonl(out / "recomposition_predictions.jsonl", rec)

    # external benchmarks (Table 15): predictions carry the original gold fields
    ext = []
    skills = {"processbench": {"LLM-based verifier": .38, "ValidLLP-style": .25, "SAFE": .31, "TP-as-a-Judge": .34,
                               "MetaPCR w/o bridge checks": .26, "Full MetaPCR-LLM": .21},
              "ragtruth": {"LLM-based verifier": .38, "ValidLLP-style": .2, "VeriCoT": .24,
                           "MetaPCR w/o bridge checks": .21, "Full MetaPCR-LLM": .16},
              "proofwriter": {"LLM-based verifier": .35, "ValidLLP-style": .28, "VeriCoT": .21,
                              "MetaPCR w/o bridge checks": .19, "Full MetaPCR-LLM": .15}}
    for task, n in (("processbench", 450), ("ragtruth", 600), ("proofwriter", 500)):
        items = []
        k = 0
        while len(items) < n:
            src = f"{task}:src{k}"
            for _ in range(1 if task == "processbench" else int(rng.integers(1, 5))):
                if len(items) >= n:
                    break
                it = {"instance_id": f"{task}:{len(items)}", "source_id": src, "u": rng.random()}
                if task == "processbench":
                    it["gold_first_error"] = -1 if rng.random() < .45 else int(rng.integers(0, 8))
                elif task == "ragtruth":
                    it["gold_halluc"] = bool(rng.random() < .42)
                else:
                    it["gold_label"] = str(rng.choice(["True", "False", "Unknown"]))
                items.append(it)
            k += 1
        for s, e in skills[task].items():
            for it in items:
                wrong = (0.75 * it["u"] + 0.25 * rng.random()) < e
                r = {k2: v for k2, v in it.items() if k2 != "u"}
                r.update(system=s, task=task, status="ok")
                if task == "processbench":
                    g = it["gold_first_error"]
                    r["pred_first_error"] = g if not wrong else (-1 if g != -1 else int(rng.integers(0, 8)))
                elif task == "ragtruth":
                    r["pred_halluc"] = it["gold_halluc"] != wrong
                else:
                    r["pred_label"] = it["gold_label"] if not wrong else str(rng.choice([x for x in ("True", "False", "Unknown") if x != it["gold_label"]]))
                ext.append(r)
    write_jsonl(out / "external_predictions.jsonl", ext)

    # composition controls (Table 17) with origins
    comp = []
    origins = [("released", 420), ("transformed", 510), ("stress", 400), ("local_pass", 150)]
    cinst = []
    for o, n in origins:
        for k in range(n):
            base = inst[int(rng.integers(len(inst)))]
            nbr = int(rng.integers(1, 4))
            bad = rng.random() < .5
            cinst.append({**base, "instance_id": f"comp:{o}:{k}", "source_id": f"comp:{o}:src{k // 2}", "regime": "mixed",
                          "origin": o, "gold_halluc": bool(bad or rng.random() < .2),
                          "bridges": [{"bridge_id": f"b{j}", "gold_valid": not (bad and j == 0)} for j in range(nbr)],
                          "u": float(rng.random())})
    cprof = {"ValidLLP-style": .25, "Native-validator portfolio": .30, "MetaPCR without bridge checks": .28,
             "Full MetaPCR-LLM": .16}
    for s, e in cprof.items():
        prof = dict(full, err=e, reg={}, bridges=s == "Full MetaPCR-LLM")
        for i in cinst:
            r = _predict(i, s, prof, rng)
            r["origin"] = i["origin"]
            if rng.random() < {"ValidLLP-style": .04, "Native-validator portfolio": .08,
                               "MetaPCR without bridge checks": .03, "Full MetaPCR-LLM": .02}[s]:
                r["pred_halluc"] = None
            comp.append(r)
    write_jsonl(out / "composition_predictions.jsonl", comp)

    # human study (Table 13)
    pyr = random.Random(seed)
    pool = [f"item{k}" for k in range(145)]
    conds = {"LLM only": .68, "+ Logic Program": .75, "+ Single-logic PCR": .79, "+ ValidLLP-style report": .83,
             "+ Full MetaPCR-LLM": .88}
    interp = {"LLM only": 3.56, "+ Logic Program": 3.7, "+ Single-logic PCR": 3.9, "+ ValidLLP-style report": 4.1,
              "+ Full MetaPCR-LLM": 4.24}
    hum = []
    for e in range(12):
        items = pyr.sample(pool, 30)
        for it in items:
            base_conf = pyr.uniform(40, 70)
            for c, a in conds.items():
                ok = pyr.random() < a
                hum.append({"evaluator": f"E{e}", "item_id": it, "condition": c, "correct": ok,
                            "judgment": "halluc" if ok else "ok",
                            "confidence_0_100": min(100, base_conf + 100 * (a - .68) * 1.2 + pyr.gauss(0, 5)),
                            "interpretability_1_5": min(5, max(1, round(interp[c] + pyr.gauss(0, .6)))),
                            "failure_loc_correct": pyr.random() < a - .1})
    write_jsonl(out / "human_study.jsonl", hum)

    # annotation: original six annotators and the 250-case two-expert audit
    ann = []
    for k in range(400):
        true = pyr.choice(REGIMES)
        for a in range(6):
            ann.append({"item_id": f"orig{k}", "annotator": f"A{a}",
                        "regime": true if pyr.random() < .93 else pyr.choice(REGIMES),
                        "bridge_required": (true == "mixed") if pyr.random() < .9 else (true != "mixed")})
    write_jsonl(out / "annotations_original.jsonl", ann)
    aud = []
    for k in range(250):
        true = pyr.choice(REGIMES)
        val = pyr.choice(VALIDATORS[:4])
        faithful = pyr.random() < {"prolog": .9, "z3": .86, "alp": .84, "problog": .82}[val]
        for a in range(2):
            aud.append({"item_id": f"aud{k}", "annotator": f"X{a}", "validator": val,
                        "regime": true if pyr.random() < .94 else pyr.choice(REGIMES),
                        "faithful": faithful if pyr.random() < .93 else not faithful})
    write_jsonl(out / "audit_250.jsonl", aud)

    # formalization attempt log (Table 22), cap k = 5
    log = []
    first = {"lean": .883, "z3": .791, "alp": .847, "problog": .803}
    rescue = {"lean": .4, "z3": .8, "alp": .35, "problog": .7}
    for v, p1 in first.items():
        for k in range(600):
            if pyr.random() < p1:
                att = [True]
            else:
                att = [False]
                for _ in range(5):
                    ok = pyr.random() < rescue[v] / 2
                    att.append(ok)
                    if ok:
                        break
            log.append({"validator": v, "obligation_id": f"{v}:{k}", "attempts": att})
    write_jsonl(out / "formalization_attempts.jsonl", log)

    # manual error analysis (Table 21) and screening ledger (Table 16)
    cats = (["Semantic decomposition"] * 28 + ["Logic routing"] * 27 + ["Formalization"] * 21 +
            ["Cross-logic bridge"] * 26 + ["Native-validator limitation"] * 22)
    write_jsonl(out / "error_analysis.jsonl", [{"instance_id": f"f{k}", "category": c} for k, c in enumerate(cats)])
    ledger = []
    for origin, (mixed, notm, amb, exc) in {"released": (420, 950, 85, 45), "transformed": (510, 40, 30, 20),
                                             "stress": (400, 0, 0, 0)}.items():
        for disp, n in (("mixed", mixed), ("not_mixed", notm), ("ambiguous", amb), ("excluded", exc)):
            ledger += [{"case_id": f"{origin}:{disp}:{k}", "origin": origin, "disposition": disp} for k in range(n)]
    write_jsonl(out / "screening_ledger.jsonl", ledger)
