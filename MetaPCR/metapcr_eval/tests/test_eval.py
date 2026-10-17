"""Unit tests on hand-checkable cases. Run: python -m pytest tests/  (or python tests/test_eval.py)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metapcr_eval import agreement as A
from metapcr_eval import diagnostics as D
from metapcr_eval import external as X
from metapcr_eval import metrics as M
from metapcr_eval import stats as ST


def rec(i, g, p, src=None, **kw):
    base = {"instance_id": str(i), "source_id": src or str(i), "gold_halluc": g, "pred_halluc": p,
            "chain_gold_valid": not g, "chain_pred_valid": not p if p is not None else None}
    base.update(kw)
    return base


def test_f1_and_accuracy():
    rows = [rec(0, True, True), rec(1, True, False), rec(2, False, True), rec(3, False, False)]
    assert M.accuracy(rows) == 0.5
    assert M.f1(rows) == 0.5
    rows.append(rec(4, True, None))          # indeterminate costs recall, is not dropped
    assert M.accuracy(rows) == 0.4
    assert abs(M.f1(rows) - 2 * 0.5 * (1 / 3) / (0.5 + 1 / 3)) < 1e-12


def test_coverage_violation_rrhr():
    rows = [rec(0, False, False, accepted=True, violates_hard_constraint=False),
            rec(1, True, False, accepted=True, violates_hard_constraint=True),
            rec(2, True, True, accepted=False, violates_hard_constraint=False)]
    assert M.coverage(rows) == 2 / 3
    assert M.violation_rate(rows) == 0.5
    assert M.rrhr(rows) == 0.5                # one of two invalid chains accepted


def test_step_bridge_localization():
    rows = [{"steps": [{"gold_valid": True, "pred_valid": True}, {"gold_valid": False, "pred_valid": None}],
             "bridges": [{"gold_valid": True, "pred_valid": False}], "gold_defect_step": "s1", "pred_defect_step": "s1"},
            {"steps": [{"gold_valid": True, "pred_valid": True}], "bridges": None,
             "gold_defect_step": None, "pred_defect_step": None}]
    assert M.step_validity_accuracy(rows) == 2 / 3
    assert M.bridge_validation_accuracy(rows) == 0.0
    assert M.failure_localization_accuracy(rows) == 1.0
    assert M.bridge_validation_accuracy([{"bridges": None}]) is None


def test_holm_is_monotone():
    adj = ST.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_bootstrap_ci_contains_diff_and_clusters():
    a, b = [], []
    for i in range(400):
        g = i % 2 == 0
        a.append(rec(i, g, g if i % 10 else not g, src=str(i // 2)))
        b.append(rec(i, g, g if i % 5 else not g, src=str(i // 2)))
    r = ST.paired_cluster_bootstrap(a, b, "accuracy", B=2000)
    assert abs(r["diff"] - 0.1) < 1e-12
    assert r["ci_low"] <= 0.1 <= r["ci_high"] and r["n_clusters"] == 200
    assert r["p_boot"] >= r["p_boot_floor"]


def test_mcnemar_refuses_clustered_units():
    a = [rec(0, True, True, src="s"), rec(1, False, False, src="s")]
    try:
        ST.mcnemar_exact(a, a)
        assert False
    except ValueError:
        pass


def test_reported_p_check_flags_table20():
    rows = ST.check_reported_p([{"name": "ProcessBench", "diff": 0.04, "ci_low": 0.018, "ci_high": 0.062,
                                 "reported_p": 1.06e-7}])
    assert not rows[0]["consistent"] and 1e-4 < rows[0]["implied_raw_p"] < 1e-3


def test_processbench_f1():
    rows = [{"gold_first_error": -1, "pred_first_error": -1}, {"gold_first_error": -1, "pred_first_error": 2},
            {"gold_first_error": 3, "pred_first_error": 3}, {"gold_first_error": 1, "pred_first_error": -1}]
    r = X.processbench_f1(rows)
    assert r["acc_correct"] == 0.5 and r["acc_erroneous"] == 0.5 and r["f1"] == 0.5


def test_manifest_is_frozen(tmp=None):
    m = X.make_manifest("pb", "v1.0", "test", [str(i) for i in range(100)], 10)
    m2 = X.make_manifest("pb", "v1.0", "test", [str(i) for i in range(100)], 10)
    assert m["ids"] == m2["ids"] and m["n"] == 10


def test_fleiss_and_cohen():
    perfect = [["a", "a"], ["b", "b"], ["a", "a"]]
    assert A.fleiss_kappa(perfect) == 1.0 and A.cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    # Fleiss (1971) style check: total disagreement on a balanced binary task
    assert A.fleiss_kappa([["a", "b"], ["b", "a"]]) < 0


def test_retry_cap_and_conditional_mean():
    log = [{"validator": "lean", "obligation_id": "1", "attempts": [True]},
           {"validator": "lean", "obligation_id": "2", "attempts": [False, False, True]}]
    r = D.formalization_acceptance(log, cap=5)[0]
    assert r["mean_retries_all"] == 1.0 and r["mean_retries_given_first_failure"] == 2.0
    try:
        D.formalization_acceptance([{"validator": "z3", "obligation_id": "x", "attempts": [False] * 7}], cap=5)
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
