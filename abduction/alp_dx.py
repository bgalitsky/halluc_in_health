from __future__ import annotations
from typing import Sequence, List
from pathlib import Path
import os

from pyswip import Prolog, Functor, Variable, Query

HERE = Path(__file__).resolve().parent
PL_FILE = HERE / "alp_dx.pl"

class AlpDiagnosis:
    def __init__(self, pl_path: str | os.PathLike = PL_FILE):
        pl_path = Path(pl_path)
        if not pl_path.exists():
            raise FileNotFoundError(f"Prolog file not found: {pl_path}")
        self.prolog = Prolog()
        self._consult(str(pl_path))

    def _consult(self, path: str):
        # normalize path for SWI on Windows
        p = path.replace("\\", "/")
        list(self.prolog.query(f'consult("{p}")'))

    def load_demo(self) -> None:
        self._call0("load_demo")

    def clear_obs(self) -> None:
        self._call0("clear_obs")

    def assume_obs(self, symptoms: Sequence[str]) -> None:
        self._call_list("assume_obs", symptoms)

    def clear_ics(self) -> None:
        self._call0("clear_ics")

    def add_ic(self, literals: Sequence[str]) -> None:
        if not literals:
            return
        body = "[" + ",".join(literals) + "]"
        self._call0(f"add_ic({body})")

    def explain_obs(self, symptoms: Sequence[str]) -> List[str]:
        return next(self.explain_obs_all(symptoms), [])

    def explain_obs_all(self, symptoms: Sequence[str]):
        syms = "[" + ",".join(self._to_atom(s) for s in symptoms) + "]"
        q = f"explain_obs({syms}, Delta)"
        for sol in self._safe_query(q):
            yield self._read_delta(sol)

    def explain_obs_k(self, symptoms: Sequence[str], k: int) -> List[str]:
        return next(self.explain_obs_k_all(symptoms, k), [])

    def explain_obs_k_all(self, symptoms: Sequence[str], k: int):
        syms = "[" + ",".join(self._to_atom(s) for s in symptoms) + "]"
        q = f"explain_obs_k({syms}, {int(k)}, Delta)"
        for sol in self._safe_query(q):
            yield self._read_delta(sol)

    # ---------------- utils ----------------

    def _call0(self, goal: str) -> bool:
        return bool(list(self.prolog.query(goal, maxresult=1)))

    def _call_list(self, pred: str, items: Sequence[str]) -> bool:
        prolog_list = "[" + ",".join(self._to_atom(s) for s in items) + "]"
        return self._call0(f"{pred}({prolog_list})")

    @staticmethod
    def _to_atom(x: str) -> str:
        s = str(x).strip().lower().replace(" ", "_")
        if not s or any(c for c in s if not (c.isalnum() or c == "_")) or s[0].isdigit():
            return f"'{s}'"
        return s

    def _read_delta(self, sol) -> List[str]:
        delta_py = sol.get("Delta")
        if isinstance(delta_py, list):
            return [str(t) for t in delta_py]
        # fallback: term_to_atom(Delta, S)
        varS = Variable()
        t2a = Functor("term_to_atom", 2)
        with self._open_query(t2a, [delta_py, varS]) as q:
            if q.nextSolution():
                return self._parse_list_atom(str(varS.value))
        return []

    @staticmethod
    def _parse_list_atom(s: str) -> List[str]:
        s = s.strip()
        if not (s.startswith("[") and s.endswith("]")):
            return [s]
        inner = s[1:-1].strip()
        return [p.strip() for p in inner.split(",")] if inner else []

    def _safe_query(self, q: str, maxresult: int | None = None):
        try:
            it = self.prolog.query(q) if maxresult is None else self.prolog.query(q, maxresult=maxresult)
            for sol in it:
                yield sol
        finally:
            pass

    from contextlib import contextmanager
    @contextmanager
    def _open_query(self, functor, args):
        q = Query(functor(*args))
        try:
            yield q
        finally:
            q.close()

    def conditional_entailment_prob(dx: AlpDiagnosis, edu, explanation: list[str], source_text: str) -> float:
        """
        P(EDU | H), approximated by NLI/entailment or logic.
        Prolog side: p_edu_given_h(ClaimAtom, HList, P).
        edu.claim_atom is used as the 'hypothesis' to be entailed under H.
        """
        if not explanation:
            return 1e-3

        h_list = "[" + ", ".join(explanation) + "]"
        query = f"p_edu_given_h({edu.claim_atom}, {h_list}, P)"

        results = list(dx.prolog.query(query, maxresult=1))
        if not results:
            return 1e-3
        return float(results[0]["P"])

    # ---------- PLP probability: plp_prob and plp_prob_single ----------

    def plp_prob(dx: AlpDiagnosis, explanation: list[str]) -> float:
        """
        Probabilistic logic program score P(H).
        Prolog side: plp_prob(HList, P).
        """
        if not explanation:
            return 1.0

        h_list = "[" + ", ".join(explanation) + "]"
        query = f"plp_prob({h_list}, P)"

        results = list(dx.prolog.query(query, maxresult=1))
        if not results:
            return 1e-6
        return float(results[0]["P"])

    def plp_prob_single(dx: AlpDiagnosis, h: str) -> float:
        """
        Probability of a single hypothesis literal h.
        Prolog side: plp_prob_single(H, P).
        """
        query = f"plp_prob_single({h}, P)"
        results = list(dx.prolog.query(query, maxresult=1))
        if not results:
            return 1e-6
        return float(results[0]["P"])

if __name__ == "__main__":
    dx = AlpDiagnosis()
    dx.load_demo()

    # Gout-like case
    s1 = ["severe_joint_pain", "swelling", "uric_acid_high", "sudden_onset_night"]
    print("Symptoms:", s1)
    print("Explanation:", dx.explain_obs(s1))  # -> ['disease(gout)']

    # RA-like case
    s2 = ["symmetric_small_joints_pain", "morning_stiffness_gt_60min", "swelling", "elevated_esr_crp"]
    print("Symptoms:", s2)
    print("Explanation:", dx.explain_obs(s2))  # -> ['disease(rheumatoid_arthritis)']
