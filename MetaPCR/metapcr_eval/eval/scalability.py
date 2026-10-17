"""Bridge-layer scalability (Section 5.14, Table 11).

A self-contained microbenchmark: native steps are pre-validated (as the paper
specifies), so it measures only registry lookup, compatibility filtering,
bridge-instance checks, structural caching and composition. Every run asserts
the analytical upper bounds of Table 11.
"""
from __future__ import annotations

import itertools
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Step:
    model: int
    pos: int
    logic: int
    conclusion: int
    premise: int
    valid: bool


def sparse_registry(n: int, density: float, rng) -> set[tuple[int, int]]:
    """Directed compatibility graph G_L over n logics; |E_L| <= n(n-1)."""
    reg = {(i, j) for i in range(n) for j in range(n) if i != j and rng.random() < density}
    for i in range(n - 1):  # keep a path so every chain logic sequence is possible
        reg |= {(i, i + 1), (i + 1, i)}
    return reg


def make_chain(L: int, n: int, q: float, rng, model: int = 0, p_valid=0.9) -> list[Step]:
    logic = int(rng.integers(n)) if n else 0
    steps = []
    for pos in range(L):
        if pos > 0 and n > 1 and rng.random() < q:
            logic = int(rng.choice([x for x in range(n) if x != logic]))
        # conclusion of pos feeds the premise of pos+1 (shared ids across models at a position)
        steps.append(Step(model, pos, logic, conclusion=pos + 1, premise=pos, valid=bool(rng.random() < p_valid)))
    return steps


class BridgeLayer:
    def __init__(self, registry, p_bridge_valid=0.92, seed=0):
        self.registry = registry
        self.rng = np.random.default_rng(seed)
        self.p = p_bridge_valid
        self.cache: dict[tuple[int, int], bool] = {}
        self.hits = self.misses = self.instances = self.crossings = self.unsupported = 0
        self.distinct: set[tuple[int, int]] = set()

    def check(self, a: Step, b: Step) -> bool:
        if a.logic == b.logic:
            return True
        key = (a.logic, b.logic)
        self.crossings += 1
        if key not in self.registry:
            self.unsupported += 1
            return False  # unsupported pair -> translation failure, not a zero-shot certificate
        if key in self.cache:      # structural translation info is cached ...
            self.hits += 1
        else:
            self.misses += 1
            self.cache[key] = True
        self.instances += 1          # ... but every instance is still semantically checked
        self.distinct.add(key)
        return bool(self.rng.random() < self.p)


def validate_chain(chain, layer) -> int:
    """Every adjacent transition is checked (native validity was established
    beforehand; an invalid native step does not hide a bridge from the count)."""
    rejected = 0
    for a, b in zip(chain, chain[1:]):
        ok_bridge = layer.check(a, b)
        if not (a.valid and b.valid and ok_bridge):
            rejected += 1
    return rejected


def recompose(chains, layer, beam: int | None):
    """Layer-aligned recomposition across K models. beam=None -> unpruned."""
    L = len(chains[0])
    candidates = 0
    frontier = [s for s in (c[0] for c in chains) if s.valid]
    if beam:
        frontier = frontier[:beam]
    rejected = 0
    reached = 0
    for pos in range(1, L):
        reached += 1
        nxt_all = [c[pos] for c in chains if c[pos].valid]        # certificate filter
        nxt = nxt_all[:beam] if beam else nxt_all
        survivors = []
        for a, b in itertools.product(frontier, nxt):
            if a.conclusion != b.premise:                         # obligation index
                continue
            candidates += 1
            if layer.check(a, b):
                survivors.append(b)
            else:
                rejected += 1
        frontier = list(dict.fromkeys(survivors))
        if beam:
            frontier = frontier[:beam]
        if not frontier:
            break
    return candidates, rejected, reached


def run_grid(ns=(2, 4, 6, 8, 10), Ls=(5, 10, 20, 40, 80), qs=(0, 0.25, 0.5, 1.0), Ks=(1, 3, 5, 8, 10),
             graphs=100, w=3, seed=42081) -> list[dict]:
    rng = np.random.default_rng(seed)
    registries = {n: sparse_registry(n, 0.3, np.random.default_rng(seed + n)) for n in ns}  # fixed per n
    rows = []
    for n, L, q, K in itertools.product(ns, Ls, qs, Ks):
        reg = registries[n]
        R = len(reg)
        assert R <= n * (n - 1)
        for strategy in ("independent", "unpruned", "beam"):
            if K == 1 and strategy != "independent":
                continue
            layer = BridgeLayer(reg, seed=int(rng.integers(1 << 31)))
            cand = rej = reached = 0
            tracemalloc.start()
            t0 = time.perf_counter()
            for _ in range(graphs):
                chains = [make_chain(L, n, q, rng, model=k) for k in range(K)]
                if strategy == "independent":
                    for c in chains:
                        cand += L - 1
                        reached += L - 1
                        rej += validate_chain(c, layer)
                else:
                    c_, r_, p_ = recompose(chains, layer, beam=w if strategy == "beam" else None)
                    cand += c_
                    rej += r_
                    reached += p_
            dt = time.perf_counter() - t0
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            per_graph_inst = layer.instances / graphs
            # ---- Table 11 bounds
            bound = {"independent": K * (L - 1), "unpruned": (L - 1) * K * K, "beam": (L - 1) * w * w}[strategy]
            assert cand / graphs <= bound + 1e-9, (strategy, n, L, q, K, cand / graphs, bound)
            assert per_graph_inst <= bound + 1e-9
            assert len(layer.distinct) <= R
            if strategy == "independent":
                assert per_graph_inst <= K * (L - 1)
            rows.append({"n": n, "L": L, "q": q, "K": K, "strategy": strategy, "registry_size": R,
                         "registry_bound": n * (n - 1),
                         "candidates_per_graph": cand / graphs, "candidate_bound": bound,
                         "candidates_per_reached_transition": cand / reached if reached else 0.0,
                         "cross_logic_transitions_per_graph": layer.crossings / graphs,
                         "unsupported_pairs_per_graph": layer.unsupported / graphs,
                         "bridge_instances_per_graph": per_graph_inst,
                         "expected_instances_independent": K * q * (L - 1) if n > 1 else 0,
                         "distinct_schemas": len(layer.distinct),
                         "cache_hit_rate": layer.hits / (layer.hits + layer.misses) if layer.hits + layer.misses else None,
                         "rejected_per_graph": rej / graphs,
                         "bridge_time_ms_per_graph": 1000 * dt / graphs, "peak_mem_kb": peak / 1024})
    return rows


def growth_exponent(rows, x: str, y: str, **fixed) -> float:
    """Log-log slope of y vs x with the other factors held fixed (empirical growth rate)."""
    sel = [r for r in rows if all(r[k] == v for k, v in fixed.items()) and r[x] > 0 and r[y] > 0]
    if len(sel) < 2:
        return float("nan")
    xs = np.log([r[x] for r in sel])
    ys = np.log([r[y] for r in sel])
    return float(np.polyfit(xs, ys, 1)[0])
