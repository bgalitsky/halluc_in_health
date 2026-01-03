"""
Compute IG* for a claim c given:
- EDUs with (optional) per-EDU information gain IG(c, e_i)
- RST rhetorical relations -> discourse weights w_i
- Abductive hypotheses Hc, each mapped to EDUs it explains
- L(Hc) estimated from web search frequency (proxy for MDL / description length)

IG*(c,S) = Σ_i w_i * IG(c,e_i)  +  λ * Σ_{h∈Hc} Σ_{e_i explained by h} w_i * ell(h)

where ell(h) is a (noisy) cost derived from web frequency:
    ell(h) = -log(freq(h) + smoothing)

This file calls Google Web Search API / Serper.dev).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
import math
import time
import json

try:
    import requests  # optional; only needed if you enable a web backend
except ImportError:
    requests = None


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class EDU:
    """Elementary Discourse Unit."""
    edu_id: str
    text: str
    role: str  # "nucleus" or "satellite" (or "unknown")
    relation: Optional[str] = None  # e.g., "Evidence", "Cause", "Background"
    ig: float = 0.0  # IG(c, e_i) provided by you or computed elsewhere


@dataclass(frozen=True)
class Hypothesis:
    """Abductive hypothesis used to support/repair entailment."""
    hyp_id: str
    query: str  # query string used for web frequency estimation
    explains_edus: List[str]  # edu_ids that rely on this hypothesis


# -----------------------------
# RST weighting
# -----------------------------

DEFAULT_RELATION_WEIGHT: Dict[str, float] = {
    # Strongly justificatory / evidential relations
    "Evidence": 1.15,
    "Justify": 1.15,
    "Cause": 1.12,
    "Result": 1.12,
    "Explanation": 1.10,
    "Elaboration": 1.00,
    "Condition": 1.05,
    "Contrast": 0.95,
    "Antithesis": 0.95,
    "Background": 0.85,
    "Example": 0.95,
}

DEFAULT_ROLE_WEIGHT: Dict[str, float] = {
    "nucleus": 1.00,
    "satellite": 0.65,
    "unknown": 0.80,
}


def compute_edu_weights(
    edus: List[EDU],
    role_weight: Dict[str, float] = DEFAULT_ROLE_WEIGHT,
    relation_weight: Dict[str, float] = DEFAULT_RELATION_WEIGHT,
    min_w: float = 0.10,
    max_w: float = 1.25,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute w_i for each EDU based on nucleus/satellite role and rhetorical relation.

    We use a simple multiplicative scheme:
        w_i = role_weight(role_i) * relation_weight(relation_i)

    Optionally normalize so mean weight = 1.0 (keeps scale stable across docs).
    """
    raw: Dict[str, float] = {}
    for e in edus:
        rw = role_weight.get(e.role.lower(), role_weight["unknown"])
        relw = 1.0
        if e.relation:
            relw = relation_weight.get(e.relation, 1.0)
        w = rw * relw
        w = max(min_w, min(max_w, w))
        raw[e.edu_id] = w

    if not normalize or len(raw) == 0:
        return raw

    mean_w = sum(raw.values()) / len(raw)
    if mean_w <= 0:
        return raw

    return {k: v / mean_w for k, v in raw.items()}


# -----------------------------
# Web frequency backends (optional)
# -----------------------------

class FrequencyCache:
    """Tiny in-memory cache with TTL."""
    def __init__(self, ttl_seconds: int = 7 * 24 * 3600):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, int]] = {}

    def get(self, key: str) -> Optional[int]:
        v = self._store.get(key)
        if not v:
            return None
        ts, freq = v
        if time.time() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return freq

    def set(self, key: str, freq: int) -> None:
        self._store[key] = (time.time(), int(freq))


def freq_serper(query: str, api_key: str, cache: Optional[FrequencyCache] = None) -> int:
    """
    Serper.dev (Google Search API) backend.

    Requires:
        pip install requests
        export SERPER_API_KEY=...

    Returns an integer "estimated total results". (Heuristic.)
    """
    if requests is None:
        raise RuntimeError("requests is not installed. pip install requests")

    cache_key = f"serper::{query}"
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": 1}

    r = requests.post(url, headers=headers, json=payload, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Serper's response can include an "searchParameters"/"results" but "totalResults" appears in some variants.
    # We'll be defensive and approximate:
    total = 0
    # Try known field names:
    for path in [("searchInformation", "totalResults"),
                 ("search_information", "total_results"),
                 ("answerBox", "totalResults")]:
        d = data
        ok = True
        for p in path:
            if isinstance(d, dict) and p in d:
                d = d[p]
            else:
                ok = False
                break
        if ok and isinstance(d, (int, float, str)):
            try:
                total = int(d)
                break
            except ValueError:
                pass

    # Fallback: if we can't find "totalResults", use number of organic results * 1000 as a crude proxy
    if total <= 0:
        organic = data.get("organic", []) if isinstance(data, dict) else []
        total = max(1, len(organic) * 1000)

    if cache:
        cache.set(cache_key, total)
    return total


def freq_bing(query: str, api_key: str, cache: Optional[FrequencyCache] = None) -> int:
    """
    Bing Web Search API backend.

    Requires:
        pip install requests
        export BING_API_KEY=...

    Returns: totalEstimatedMatches (int).
    """
    if requests is None:
        raise RuntimeError("requests is not installed. pip install requests")

    cache_key = f"bing::{query}"
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": 1, "textDecorations": False, "textFormat": "Raw"}

    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    total = int(data.get("webPages", {}).get("totalEstimatedMatches", 1))
    total = max(1, total)

    if cache:
        cache.set(cache_key, total)
    return total


# -----------------------------
# MDL / description-length proxy
# -----------------------------

def ell_from_frequency(freq: int, smoothing: float = 1.0, log_base: float = math.e) -> float:
    """
    Convert web frequency into a "cost" ell(h) ~ description length proxy.
    A simple robust choice:
        ell = -log(freq + smoothing)

    Larger freq -> lower cost; smaller freq -> higher cost.
    """
    f = max(0, int(freq))
    val = f + smoothing
    if val <= 0:
        val = smoothing
    ln = math.log(val)
    if log_base != math.e:
        ln = ln / math.log(log_base)
    return -ln


# -----------------------------
# IG* computation
# -----------------------------

def compute_ig_star(
    edus: List[EDU],
    hypotheses: List[Hypothesis],
    lam: float,
    edu_weights: Optional[Dict[str, float]] = None,
    freq_fn: Optional[Callable[[str], int]] = None,
    smoothing: float = 1.0,
    log_base: float = math.e,
    default_hyp_cost: float = 5.0,
) -> Dict[str, float]:
    """
    Returns a dict with:
      - IG_weighted
      - L_weighted
      - IG_star

    Parameters
    ----------
    lam: λ weighting for abductive complexity.
    edu_weights: optional precomputed {edu_id: w_i}. If None, computed from RST metadata.
    freq_fn: function(query)->frequency. If None, uses default_hyp_cost for ell(h).
    """
    if edu_weights is None:
        edu_weights = compute_edu_weights(edus)

    # Weighted information gain
    ig_weighted = 0.0
    for e in edus:
        w = edu_weights.get(e.edu_id, 1.0)
        ig_weighted += w * float(e.ig)

    # Discourse-weighted abductive complexity
    L_weighted = 0.0
    for h in hypotheses:
        if freq_fn is None:
            ell = float(default_hyp_cost)
        else:
            freq = freq_fn(h.query)
            ell = ell_from_frequency(freq, smoothing=smoothing, log_base=log_base)

        # Sum over EDUs the hypothesis is used to explain
        for edu_id in h.explains_edus:
            w = edu_weights.get(edu_id, 1.0)
            L_weighted += w * ell

    ig_star = ig_weighted + float(lam) * L_weighted
    return {"IG_weighted": ig_weighted, "L_weighted": L_weighted, "IG_star": ig_star}


# -----------------------------
# Example usage
# -----------------------------

def main():
    # Example EDUs (you would typically load this from your RST parser output)
    edus = [
        EDU("e1", "Patient has fever and rash.", role="nucleus", relation="Evidence", ig=0.9),
        EDU("e2", "Therefore it must be an allergic reaction.", role="nucleus", relation="Cause", ig=1.4),
        EDU("e3", "The rash appeared after new medication.", role="satellite", relation="Background", ig=0.4),
    ]

    # Example hypotheses (Hc) explaining portions of the text/claim
    hypotheses = [
        Hypothesis("h1", query='"fever rash" allergy reaction mechanism', explains_edus=["e2"]),
        Hypothesis("h2", query='"new medication" rash temporal association', explains_edus=["e3"]),
    ]

    # Choose a web backend by providing a freq_fn(query)->int
    cache = FrequencyCache()

    # Serper.dev (Google) - requires API key
    serper_key = os.environ["SERPER_API_KEY"]
    freq_fn = lambda q: freq_serper(q, serper_key, cache=cache)

    # OPTION B: Bing Web Search - requires API key
    # bing_key = os.environ["BING_API_KEY"]
    # freq_fn = lambda q: freq_bing(q, bing_key, cache=cache)

    # OPTION C: Offline stub (useful for tests / unit tests)
    stub_freqs = {
        '"fever rash" allergy reaction mechanism': 250000,
        '"new medication" rash temporal association': 1200000,
    }
    freq_fn = lambda q: stub_freqs.get(q, 1000)

    lam = 0.5
    edu_w = compute_edu_weights(edus)  # derived from role + relation

    result = compute_ig_star(
        edus=edus,
        hypotheses=hypotheses,
        lam=lam,
        edu_weights=edu_w,
        freq_fn=freq_fn,
        smoothing=1.0,
        log_base=math.e,
    )

    print("EDU weights:", json.dumps(edu_w, indent=2))
    print("Result:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
