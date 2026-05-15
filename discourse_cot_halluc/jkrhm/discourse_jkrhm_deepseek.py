import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Discourse tree structure
# ============================================================

@dataclass
class DiscourseNode:
    text: str
    role: str = "nucleus"          # nucleus / satellite
    relation: str = "span"         # evidence / contrast / elaboration / cause / etc.
    children: Optional[List["DiscourseNode"]] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


# ============================================================
# Local DeepSeek / HuggingFace geometry extractor
# ============================================================

class LocalLLMGeometry:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 1024,
        use_float16: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" and use_float16 else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        if self.device == "cpu":
            self.model.to(self.device)

        self.model.eval()

    @torch.no_grad()
    def get_last_hidden_mean(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)

        outputs = self.model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
        mask = inputs["attention_mask"].unsqueeze(-1)

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled.float()

    @torch.no_grad()
    def compute_feature_kernel(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Fast NTK-like proxy:

            K_ij = h_i dot h_j

        where h_i is mean pooled last-layer hidden state.
        """
        H = self.get_last_hidden_mean(texts)

        if normalize:
            H = torch.nn.functional.normalize(H, dim=-1)

        K = H @ H.T
        return K.cpu().numpy()

    @torch.no_grad()
    def _next_token_logits_from_embedding(
        self,
        input_ids: torch.Tensor,
        perturb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb_layer = self.model.get_input_embeddings()
        inputs_embeds = emb_layer(input_ids)

        if perturb is not None:
            inputs_embeds = inputs_embeds + perturb

        outputs = self.model(inputs_embeds=inputs_embeds)
        logits = outputs.logits[:, -1, :]

        return logits.float()

    def estimate_sigma_finite_difference(
        self,
        text: str,
        n_trials: int = 8,
        eps: float = 1e-3,
    ) -> float:
        """
        Estimates sigma_max by perturbing input embeddings and measuring:

            ||delta logits|| / ||delta embeddings||

        This is a practical proxy for local decoder sensitivity.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)

        input_ids = inputs["input_ids"]

        with torch.no_grad():
            base_logits = self._next_token_logits_from_embedding(input_ids)

            emb_layer = self.model.get_input_embeddings()
            base_emb = emb_layer(input_ids)

            ratios = []

            for _ in range(n_trials):
                delta = torch.randn_like(base_emb)
                delta = eps * delta / (delta.norm() + 1e-8)

                pert_logits = self._next_token_logits_from_embedding(
                    input_ids=input_ids,
                    perturb=delta,
                )

                output_diff = (pert_logits - base_logits).norm()
                input_diff = delta.norm()

                ratios.append((output_diff / input_diff).item())

        return max(ratios)


# ============================================================
# Discourse-JKRHM scorer
# ============================================================

class DiscourseJKRHM:
    """
    Discourse-augmented Joint Knowledge–Reasoning Hallucination Measure.

    Geometry risk:

        -log det(K) + log sigma_max + 2 log kappa(K)

    Full risk:

        Risk(u_h, T) =
            geometry_risk
            + lambda_counter * D_counter
            + mu_disc * (1 - DiscScore(T))

    Higher = more likely hallucination.
    """

    def __init__(
        self,
        lambda_counter: float = 1.0,
        mu_disc: float = 1.0,
        eps: float = 1e-8,
    ):
        self.lambda_counter = lambda_counter
        self.mu_disc = mu_disc
        self.eps = eps

    def kernel_stats(self, K: np.ndarray) -> Dict[str, object]:
        K = np.asarray(K, dtype=np.float64)

        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("K must be a square matrix.")

        K = K + self.eps * np.eye(K.shape[0])

        eigvals = np.linalg.eigvalsh(K)
        eigvals = np.clip(eigvals, self.eps, None)

        logdet = float(np.sum(np.log(eigvals)))
        kappa = float(np.max(eigvals) / np.min(eigvals))

        return {
            "logdet_K": logdet,
            "kappa_K": kappa,
            "eigvals": eigvals,
        }

    def geometry_risk(self, K: np.ndarray, sigma_max: float) -> Dict[str, float]:
        stats = self.kernel_stats(K)

        logdet = stats["logdet_K"]
        kappa = stats["kappa_K"]
        log_sigma = math.log(max(float(sigma_max), self.eps))

        risk = -logdet + log_sigma + 2.0 * math.log(max(kappa, self.eps))

        return {
            "logdet_K": logdet,
            "kappa_K": kappa,
            "log_sigma_max": log_sigma,
            "geometry_risk": float(risk),
        }

    def flatten_tree(self, root: DiscourseNode) -> List[DiscourseNode]:
        nodes = [root]
        for child in root.children:
            nodes.extend(self.flatten_tree(child))
        return nodes

    def tree_depth(self, root: DiscourseNode) -> int:
        if not root.children:
            return 1
        return 1 + max(self.tree_depth(child) for child in root.children)

    def count_edges(self, root: DiscourseNode) -> int:
        return len(root.children) + sum(self.count_edges(child) for child in root.children)

    def relation_entropy(self, nodes: List[DiscourseNode]) -> float:
        relations = [node.relation.lower() for node in nodes if node.relation]
        counts = Counter(relations)
        total = sum(counts.values())

        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(p + self.eps)

        max_entropy = math.log(len(counts) + self.eps)

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def nucleus_satellite_ratio(self, nodes: List[DiscourseNode]) -> float:
        nuclei = sum(1 for node in nodes if node.role.lower() == "nucleus")
        satellites = sum(1 for node in nodes if node.role.lower() == "satellite")
        return nuclei / max(satellites, 1)

    def contrast_ratio(self, nodes: List[DiscourseNode]) -> float:
        contrastive = {
            "contrast",
            "antithesis",
            "concession",
            "condition",
            "otherwise",
            "alternative",
            "counterexample",
        }

        if not nodes:
            return 0.0

        return sum(1 for node in nodes if node.relation.lower() in contrastive) / len(nodes)

    def evidence_ratio(self, nodes: List[DiscourseNode]) -> float:
        evidential = {
            "evidence",
            "justify",
            "justification",
            "explanation",
            "cause",
            "reason",
            "support",
            "result",
        }

        if not nodes:
            return 0.0

        return sum(1 for node in nodes if node.relation.lower() in evidential) / len(nodes)

    def discourse_score(self, root: DiscourseNode) -> Dict[str, float]:
        nodes = self.flatten_tree(root)

        n_nodes = len(nodes)
        n_edges = self.count_edges(root)
        depth = self.tree_depth(root)
        branching = n_edges / max(n_nodes, 1)

        ns_ratio = self.nucleus_satellite_ratio(nodes)
        ns_balance = math.exp(-abs(math.log(max(ns_ratio, self.eps))))

        rel_entropy = self.relation_entropy(nodes)
        evidence = self.evidence_ratio(nodes)
        contrast = self.contrast_ratio(nodes)

        depth_score = min(depth / 6.0, 1.0)
        branching_score = min(branching, 1.0)

        disc_score = (
            0.20 * depth_score
            + 0.15 * branching_score
            + 0.20 * ns_balance
            + 0.20 * rel_entropy
            + 0.15 * evidence
            + 0.10 * contrast
        )

        disc_score = float(np.clip(disc_score, 0.0, 1.0))

        return {
            "n_nodes": float(n_nodes),
            "n_edges": float(n_edges),
            "depth": float(depth),
            "branching": float(branching),
            "nucleus_satellite_ratio": float(ns_ratio),
            "ns_balance": float(ns_balance),
            "relation_entropy": float(rel_entropy),
            "evidence_ratio": float(evidence),
            "contrast_ratio": float(contrast),
            "discourse_score": float(disc_score),
            "discourse_risk": float(1.0 - disc_score),
        }

    def counter_abduction_score(
        self,
        main_hypothesis_ig: float,
        rival_hypotheses: List[Tuple[float, float]],
    ) -> float:
        """
        D_counter(H0) = max_i [IG(H_i) - IG(H0) - Cost(H_i)].

        rival_hypotheses = [(rival_information_gain, rival_cost), ...]
        """
        if not rival_hypotheses:
            return 0.0

        defeats = [
            rival_ig - main_hypothesis_ig - rival_cost
            for rival_ig, rival_cost in rival_hypotheses
        ]

        return float(max(0.0, max(defeats)))

    def compute(
        self,
        K: np.ndarray,
        sigma_max: float,
        discourse_tree: DiscourseNode,
        main_hypothesis_ig: float = 0.0,
        rival_hypotheses: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, float]:

        if rival_hypotheses is None:
            rival_hypotheses = []

        geom = self.geometry_risk(K, sigma_max)
        disc = self.discourse_score(discourse_tree)

        d_counter = self.counter_abduction_score(
            main_hypothesis_ig=main_hypothesis_ig,
            rival_hypotheses=rival_hypotheses,
        )

        total_risk = (
            geom["geometry_risk"]
            + self.lambda_counter * d_counter
            + self.mu_disc * disc["discourse_risk"]
        )

        result = {}
        result.update(geom)
        result.update(disc)
        result["D_counter"] = d_counter
        result["discourse_jkrhm_risk"] = float(total_risk)

        return result


# ============================================================
# Demo discourse tree
# ============================================================

def build_demo_discourse_tree() -> DiscourseNode:
    return DiscourseNode(
        text="The patient likely has flu.",
        role="nucleus",
        relation="claim",
        children=[
            DiscourseNode(
                text="The patient has fever and cough.",
                role="satellite",
                relation="evidence",
            ),
            DiscourseNode(
                text="These symptoms are common in flu.",
                role="satellite",
                relation="elaboration",
            ),
        ],
    )


# ============================================================
# Main runnable example
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/deepseek-llm-7b-base",
        help="HuggingFace model name or local model path.",
    )

    parser.add_argument(
        "--no_model",
        action="store_true",
        help="Use toy K and sigma instead of loading DeepSeek/local LLM.",
    )

    parser.add_argument(
        "--lambda_counter",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--mu_disc",
        type=float,
        default=2.0,
    )

    args = parser.parse_args()

    texts = [
        "The patient likely has flu because fever and cough are present.",
        "The patient may have meningitis because fever, neck stiffness, and photophobia are present.",
        "Antibiotics are needed because all fevers are bacterial.",
    ]

    generated_explanation = texts[0]
    discourse_tree = build_demo_discourse_tree()

    if args.no_model:
        K = np.array([
            [1.00, 0.70, 0.50],
            [0.70, 1.00, 0.45],
            [0.50, 0.45, 1.00],
        ])

        sigma_max = 1.15

    else:
        extractor = LocalLLMGeometry(args.model)
        K = extractor.compute_feature_kernel(texts)
        sigma_max = extractor.estimate_sigma_finite_difference(
            generated_explanation,
            n_trials=8,
            eps=1e-3,
        )

    # Main hypothesis: flu
    main_ig = 0.55

    # Rival hypotheses: meningitis and pneumonia
    rivals = [
        (0.90, 0.15),
        (0.60, 0.10),
    ]

    scorer = DiscourseJKRHM(
        lambda_counter=args.lambda_counter,
        mu_disc=args.mu_disc,
    )

    scores = scorer.compute(
        K=K,
        sigma_max=sigma_max,
        discourse_tree=discourse_tree,
        main_hypothesis_ig=main_ig,
        rival_hypotheses=rivals,
    )

    print("\n=== Discourse-JKRHM Results ===")
    for key, value in scores.items():
        print(f"{key:32s}: {value:.6f}")


if __name__ == "__main__":
    main()