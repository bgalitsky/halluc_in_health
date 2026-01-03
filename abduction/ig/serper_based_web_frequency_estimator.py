from typing import Optional
import time
import math

try:
    import requests
except ImportError:
    requests = None


class SerperFrequencyEstimator:
    """
    Web-frequency estimator using Serper.dev (Google Search API).

    Usage:
        cache = FrequencyCache(ttl_seconds=7*24*3600)
        freq_est = SerperFrequencyEstimator(api_key=SERPER_API_KEY, cache=cache)
        freq = freq_est("fever rash allergy mechanism")
    """

    SERPER_URL = "https://google.serper.dev/search"

    def __init__(
        self,
        api_key: str,
        cache=None,
        default_freq: int = 1,
        timeout: int = 20,
        min_freq: int = 1,
        max_freq: Optional[int] = None,
    ):
        if requests is None:
            raise RuntimeError("requests is required. pip install requests")

        self.api_key = api_key
        self.cache = cache
        self.default_freq = default_freq
        self.timeout = timeout
        self.min_freq = min_freq
        self.max_freq = max_freq

    def __call__(self, query: str) -> int:
        """
        Returns an integer proxy for web frequency.
        Can be passed directly as freq_fn to IG* computation.
        """
        cache_key = f"serper::{query}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "num": 1,
        }

        try:
            r = requests.post(
                self.SERPER_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            freq = self._extract_frequency(data)
        except Exception:
            freq = self.default_freq

        freq = max(self.min_freq, int(freq))
        if self.max_freq is not None:
            freq = min(freq, self.max_freq)

        if self.cache:
            self.cache.set(cache_key, freq)

        return freq

    def _extract_frequency(self, data: dict) -> int:
        """
        Attempt to extract total result count from Serper response.
        Falls back to a conservative heuristic if unavailable.
        """
        # Known patterns (Serper response schema may evolve)
        candidates = [
            ("searchInformation", "totalResults"),
            ("search_information", "total_results"),
            ("answerBox", "totalResults"),
        ]

        for path in candidates:
            d = data
            ok = True
            for p in path:
                if isinstance(d, dict) and p in d:
                    d = d[p]
                else:
                    ok = False
                    break
            if ok:
                try:
                    return int(d)
                except Exception:
                    pass

        # Fallback heuristic:
        # number of organic results × constant
        organic = data.get("organic", [])
        if isinstance(organic, list) and len(organic) > 0:
            return len(organic) * 1000

        return self.default_freq
