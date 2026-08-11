from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple
import json, re


class LLM(Protocol):
    def __call__(self, prompt: str) -> str: ...


class NLIScorer(Protocol):
    def __call__(self, premise: str, hypothesis: str) -> Dict[str, float]: ...


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> Sequence[Tuple[str, float]]: ...


FeatureFn = Callable[[str, Sequence[str]], float]
EvidenceOnlyFeatureFn = Callable[[Sequence[str]], float]
DiscourseWeightFn = Callable[[str, str, Sequence[str]], float]
DefeatFn = Callable[[str, str, str, Sequence[str], Sequence[str]], float]


@dataclass
class RhetoricWeights:
    relevance: float = 1/3
    coherence: float = 1/3
    coverage: float = 1/3
    tau: float = 0.60


@dataclass
class ValidationWeights:
    explanatory_adequacy: float = 1/3
    grounding: float = 1/3
    discourse: float = 1/3
    lambda_contradiction: float = 0.50
    lambda_defeat: float = 0.50


@dataclass
class CandidateHypothesis:
    premise: str
    provisional_answer: str = ""


@dataclass
class CandidateEvaluation:
    premise: str
    provisional_answer: str
    goal_fit: float
    contradiction: float
    defeat: float
    explanatory_adequacy_raw: float
    explanatory_adequacy: float
    grounding: float
    discourse_weight: float
    final_score: float
    retrieved_support: List[Tuple[str, float]]


@dataclass
class RunResult:
    answer: str
    mode: str
    rhetoric_sufficiency: float
    rhetoric_features: Dict[str, float]
    selected_premise: Optional[str]
    candidates: List[CandidateEvaluation]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_simplex(values):
    vals = [max(0.0, float(v)) for v in values]
    s = sum(vals)
    return [1/len(vals)] * len(vals) if s <= 0 else [v/s for v in vals]


def evidence_text(evidence: Sequence[str]) -> str:
    return "\n".join(f"[E{i+1}] {e}" for i, e in enumerate(evidence))


def lexical_relevance(query: str, evidence: Sequence[str]) -> float:
    q = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
    if not q:
        return 0.0
    e = set(re.findall(r"[A-Za-z0-9_]+", " ".join(evidence).lower()))
    return clamp01(len(q & e) / len(q))


def lexical_coverage(query: str, evidence: Sequence[str]) -> float:
    stop = {"the","a","an","of","to","in","on","for","and","or","is","are",
            "was","were","be","with","what","which","who","how","why"}
    q = {t for t in re.findall(r"[A-Za-z0-9_]+", query.lower())
         if t not in stop and len(t) > 2}
    if not q:
        return 1.0
    e = set(re.findall(r"[A-Za-z0-9_]+", " ".join(evidence).lower()))
    return clamp01(len(q & e) / len(q))


def lexical_coherence(evidence: Sequence[str]) -> float:
    if len(evidence) <= 1:
        return 1.0 if evidence else 0.0
    vals = []
    for a, b in zip(evidence[:-1], evidence[1:]):
        ta = set(re.findall(r"[A-Za-z0-9_]+", a.lower()))
        tb = set(re.findall(r"[A-Za-z0-9_]+", b.lower()))
        u = ta | tb
        vals.append(len(ta & tb)/len(u) if u else 0.0)
    return clamp01(sum(vals)/len(vals))


def default_discourse_weight(premise, query, evidence):
    return 0.5


def default_defeat(premise, provisional_answer, query, evidence, rivals):
    return 0.0


class RhetoricSufficiencyEstimator:
    def __init__(self, weights: RhetoricWeights,
                 relevance_fn: FeatureFn = lexical_relevance,
                 coherence_fn: EvidenceOnlyFeatureFn = lexical_coherence,
                 coverage_fn: FeatureFn = lexical_coverage):
        self.weights = weights
        self.relevance_fn = relevance_fn
        self.coherence_fn = coherence_fn
        self.coverage_fn = coverage_fn

    def features(self, query, evidence):
        return {
            "relevance": clamp01(self.relevance_fn(query, evidence)),
            "coherence": clamp01(self.coherence_fn(evidence)),
            "coverage": clamp01(self.coverage_fn(query, evidence)),
        }

    def score(self, query, evidence):
        f = self.features(query, evidence)
        w = normalize_simplex([self.weights.relevance,
                               self.weights.coherence,
                               self.weights.coverage])
        s = w[0]*f["relevance"] + w[1]*f["coherence"] + w[2]*f["coverage"]
        return clamp01(s), f


class AbductiveHypothesisGenerator:
    def __init__(self, llm: LLM, max_candidates: int = 6):
        self.llm = llm
        self.max_candidates = max_candidates

    def generate(self, query, evidence):
        prompt = f"""You are the hypothesis-generation stage of an abductive RAG system.

QUERY:
{query}

RETRIEVED EVIDENCE:
{evidence_text(evidence)}

Generate at most {self.max_candidates} minimal missing premises that could make
a coherent answer possible. A premise is a hypothesis, not an observed fact.
Generate alternatives when more than one explanation is plausible.

Return ONLY JSON:
[
  {{"premise":"...", "provisional_answer":"..."}}
]
"""
        raw = self.llm(prompt).strip()
        try:
            items = json.loads(raw)
        except Exception:
            m = re.search(r"\[[\s\S]*\]", raw)
            items = json.loads(m.group(0)) if m else []

        out, seen = [], set()
        for item in items:
            p = str(item.get("premise","")).strip()
            a = str(item.get("provisional_answer","")).strip()
            if p and p.lower() not in seen:
                seen.add(p.lower())
                out.append(CandidateHypothesis(p, a))
            if len(out) >= self.max_candidates:
                break
        return out


class CandidateValidator:
    def __init__(self, nli: NLIScorer, retriever: Retriever,
                 weights: ValidationWeights,
                 discourse_weight_fn: DiscourseWeightFn = default_discourse_weight,
                 defeat_fn: DefeatFn = default_defeat,
                 retrieval_k: int = 5):
        self.nli = nli
        self.retriever = retriever
        self.weights = weights
        self.discourse_weight_fn = discourse_weight_fn
        self.defeat_fn = defeat_fn
        self.retrieval_k = retrieval_k

    def goal_fit(self, evidence, premise, answer):
        if not answer:
            return 0.0
        ctx = evidence_text(list(evidence)+[f"HYPOTHESIZED PREMISE: {premise}"])
        return clamp01(self.nli(ctx, answer).get("entailment", 0.0))

    def contradiction(self, evidence, premise):
        if not evidence:
            return 0.0
        return max(clamp01(self.nli(e, premise).get("contradiction",0.0))
                   for e in evidence)

    def grounding(self, premise):
        hits = list(self.retriever.search(premise, k=self.retrieval_k))
        if not hits:
            return 0.0, []
        vals = [clamp01(s) for _, s in hits]
        return clamp01(sum(vals)/len(vals)), hits

    def evaluate(self, candidate, query, evidence, rivals):
        gf = self.goal_fit(evidence, candidate.premise, candidate.provisional_answer)
        con = self.contradiction(evidence, candidate.premise)
        defeat = clamp01(self.defeat_fn(candidate.premise,
                                        candidate.provisional_answer,
                                        query, evidence, rivals))

        raw = (gf
               - self.weights.lambda_contradiction * con
               - self.weights.lambda_defeat * defeat)

        lo = -(self.weights.lambda_contradiction + self.weights.lambda_defeat)
        adequacy = clamp01((raw - lo) / (1.0 - lo))

        grounding, hits = self.grounding(candidate.premise)
        dw = clamp01(self.discourse_weight_fn(candidate.premise, query, evidence))

        w = normalize_simplex([self.weights.explanatory_adequacy,
                               self.weights.grounding,
                               self.weights.discourse])
        score = clamp01(w[0]*adequacy + w[1]*grounding + w[2]*dw)

        return CandidateEvaluation(
            candidate.premise, candidate.provisional_answer,
            gf, con, defeat, raw, adequacy, grounding, dw, score, hits
        )


class IntraLLMAbductiveRAG:
    def __init__(self, llm: LLM,
                 sufficiency: RhetoricSufficiencyEstimator,
                 generator: AbductiveHypothesisGenerator,
                 validator: CandidateValidator,
                 min_candidate_score: float = 0.55):
        self.llm = llm
        self.sufficiency = sufficiency
        self.generator = generator
        self.validator = validator
        self.min_candidate_score = min_candidate_score

    def run(self, query: str, evidence: Sequence[str]) -> RunResult:
        rs, rf = self.sufficiency.score(query, evidence)

        if rs >= self.sufficiency.weights.tau:
            prompt = f"""Answer using only the evidence. If it is insufficient, say so.

QUERY:
{query}

EVIDENCE:
{evidence_text(evidence)}
"""
            return RunResult(self.llm(prompt).strip(), "direct_rag",
                             rs, rf, None, [])

        candidates = self.generator.generate(query, evidence)
        if not candidates:
            return RunResult("Insufficient evidence and no abductive premise generated.",
                             "abstain", rs, rf, None, [])

        premises = [c.premise for c in candidates]
        evs = []
        for c in candidates:
            rivals = [p for p in premises if p != c.premise]
            evs.append(self.validator.evaluate(c, query, evidence, rivals))
        evs.sort(key=lambda x: x.final_score, reverse=True)

        best = evs[0]
        if best.final_score < self.min_candidate_score:
            return RunResult("Insufficient evidence: no premise passed validation.",
                             "abstain", rs, rf, None, evs)

        prompt = f"""Answer using the evidence and the explicitly marked abductive premise.
The premise is hypothesized, not observed. Preserve uncertainty.

QUERY:
{query}

EVIDENCE:
{evidence_text(evidence)}

ABDUCTIVE PREMISE:
{best.premise}
"""
        ans = self.llm(prompt).strip()
        return RunResult(ans, "abductive_rag", rs, rf, best.premise, evs)


def load_weight_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    r, v = cfg["rhetoric"], cfg["validation"]
    return (
        RhetoricWeights(r["relevance"], r["coherence"], r["coverage"], r["tau"]),
        ValidationWeights(v["explanatory_adequacy"], v["grounding"], v["discourse"],
                          v.get("lambda_contradiction",0.5),
                          v.get("lambda_defeat",0.5))
    )
