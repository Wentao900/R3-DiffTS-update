"""Learned per-sample text validity.

The module maps raw features to a gate in (0, 1). It replaces the hand-written
quality product and the controller text_gate as the value that sets how far
pattern routing moves from the numeric logits toward the text logits.

No trust weight, alignment cap, source mix, or clamp floor is applied here.
The final linear layers are zero-initialized, so a fresh gate is sigmoid(0) = 0.5
until training moves it. That is an initialization, not a validity coefficient.
"""

import torch
import torch.nn as nn

from utils.text_signals import VALIDITY_FEATURE_DIM

POLICY_CONDITION_BASE_DIM = 25
VALIDITY_CONDITION_EXTRA_DIM = 4


def benefit_gate_target(err_numeric: torch.Tensor, err_text: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Map a pair of forecast errors to a gate label in (0, 1).

    The label is err_numeric / (err_numeric + err_text). It is above 0.5 when
    text routing reduces error and near 0 when text routing is worse. `eps`
    only keeps the ratio defined when both errors are zero; it is not a
    validity weight. Callers must detach this target. It is never a model input.
    """
    numeric = err_numeric.clamp(min=0.0)
    text = err_text.clamp(min=0.0)
    return numeric / (numeric + text + float(eps))


class TextValidityModule(nn.Module):
    def __init__(self, text_dim: int, proj_dim: int = 32, hidden_dim: int = 64, evidence_dim: int = VALIDITY_FEATURE_DIM, stats_dim: int = 8):
        super().__init__()
        self.proj_dim = int(proj_dim)
        self.evidence_dim = int(evidence_dim)
        self.stats_dim = int(stats_dim)
        self.source_proj = nn.Linear(int(text_dim), self.proj_dim)
        in_dim = 3 * self.proj_dim + 3 + self.stats_dim + self.evidence_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.source_head = nn.Linear(int(hidden_dim), 3)
        self.sample_head = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.source_head.weight)
        nn.init.zeros_(self.source_head.bias)
        nn.init.zeros_(self.sample_head.weight)
        nn.init.zeros_(self.sample_head.bias)
        self.register_buffer("gate_sum", torch.zeros(()))
        self.register_buffer("gate_count", torch.zeros(()))

    def forward(self, source_emb, source_mask, stats, evidence, disagreement):
        """
        source_emb: (B, 3, D) frozen text embeddings for raw / retrieved / CoT
        source_mask: (B, 3) 1 when that source string is non-empty
        stats: (B, stats_dim) numeric history descriptors
        evidence: (B, evidence_dim) raw features, not pre-multiplied scores
        disagreement: (B, 2) KL(numeric || ungated text) and KL(ungated text || numeric)
        """
        mask = source_mask.float()
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        projected = self.source_proj(source_emb.float())
        projected = projected * mask.unsqueeze(-1)
        flat = projected.reshape(projected.shape[0], -1)
        stats = stats.float()
        evidence = evidence.float()
        disagreement = disagreement.float()
        hidden = self.mlp(torch.cat([flat, mask, stats, evidence, disagreement], dim=-1))
        source_gates = torch.sigmoid(self.source_head(hidden)) * mask
        sample_gate = torch.sigmoid(self.sample_head(hidden)).squeeze(-1)
        return sample_gate, source_gates

    @torch.no_grad()
    def update_running_mean(self, sample_gate: torch.Tensor) -> None:
        self.gate_sum += sample_gate.detach().float().sum()
        self.gate_count += float(sample_gate.shape[0])

    def running_mean(self, fallback: torch.Tensor) -> torch.Tensor:
        if float(self.gate_count.item()) <= 0:
            return fallback.detach()
        return (self.gate_sum / self.gate_count).to(device=fallback.device, dtype=fallback.dtype)
