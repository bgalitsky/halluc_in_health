# Save this file to FCA/core_fca.py
from fractions import Fraction
from typing import Dict, FrozenSet, Set, Tuple, List


class FormalContextEngine:
    def __init__(self, context: Dict[str, Set[str]], attribute_weights: Dict[str, float] = None):
        self.context = {k: frozenset(v) for k, v in context.items()}
        self.attributes = frozenset().union(*self.context.values())
        self.weights = attribute_weights if attribute_weights else {a: 1.0 for a in self.attributes}

    def get_extent(self, intent: FrozenSet[str]) -> FrozenSet[str]:
        return frozenset(obj for obj, attrs in self.context.items() if intent.issubset(attrs))

    def get_intent(self, extent: FrozenSet[str]) -> FrozenSet[str]:
        if not extent:
            return self.attributes
        return frozenset.intersection(*(self.context[obj] for obj in extent))

    def get_closure(self, intent: FrozenSet[str]) -> FrozenSet[str]:
        return self.get_intent(self.get_extent(intent))

    def get_weight(self, intent: FrozenSet[str]) -> float:
        return sum(self.weights.get(a, 1.0) for a in intent)

    def jaccard_similarity(self, x: FrozenSet[str], y: FrozenSet[str]) -> float:
        union = x | y
        if not union:
            return 1.0
        return self.get_weight(x & y) / self.get_weight(union)

    def calculate_maximum_anchor(self, candidate_intent: FrozenSet[str]) -> Tuple[str, FrozenSet[str], float]:
        if not self.context:
            return "", frozenset(), 0.0

        best_obj = ""
        best_core = frozenset()
        max_weight = -1.0
        max_sim = -1.0

        for obj, attrs in sorted(self.context.items()):
            core = candidate_intent & attrs
            weight = self.get_weight(core)
            sim = self.jaccard_similarity(candidate_intent, attrs)

            if (weight > max_weight) or (weight == max_weight and sim > max_sim):
                max_weight = weight
                max_sim = sim
                best_obj = obj
                best_core = core

        return best_obj, best_core, max_weight

    def calculate_supported_novelty(self, candidate_intent: FrozenSet[str], s_min: int = 1) -> float:
        import collections
        intents_map = collections.defaultdict(set)

        # Enumerate all subsets to find concepts
        from itertools import chain, combinations
        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

        supported_intents = []
        for st in powerset(self.attributes):
            intent_cand = frozenset(st)
            ext = self.get_extent(intent_cand)
            if len(ext) >= s_min:
                # Ensure it's a closed intent
                closed = self.get_intent(ext)
                if closed not in supported_intents:
                    supported_intents.append(closed)

        if not supported_intents:
            return 1.0

        max_sim = max(self.jaccard_similarity(candidate_intent, b) for b in supported_intents)
        return 1.0 - max_sim