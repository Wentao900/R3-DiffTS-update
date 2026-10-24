"""Four-parameter text validity gate and its stop-gradient benefit label.

g = 1[text present] * sigmoid(a * s + b + sum_s beta_s * m_s)

The benefit label is the detached inverse-variance weight
y = stopgrad(ell_null / (ell_null + ell_cond + eps)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextValidityGate(nn.Module):
    def __init__(self, n_sources=2):
        super().__init__()
        if int(n_sources) < 1:
            raise ValueError("n_sources must be positive")
        self.n_sources = int(n_sources)
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(self.n_sources))

    def forward(self, similarity, source_mask, any_mask=None):
        """
        similarity: (B,) cosine, already detached by the caller
        source_mask: (B, n_sources) in {0, 1}
        any_mask: (B,) 1 when at least one source is present
        """
        similarity = similarity.reshape(-1).float()
        source_mask = source_mask.reshape(similarity.shape[0], self.n_sources).to(
            device=similarity.device, dtype=similarity.dtype
        )
        if any_mask is None:
            any_mask = (source_mask.sum(dim=-1) > 0).to(dtype=similarity.dtype)
        else:
            any_mask = any_mask.reshape(-1).to(device=similarity.device, dtype=similarity.dtype)
        logit = self.a * similarity + self.b + (source_mask * self.beta.view(1, -1)).sum(dim=-1)
        return any_mask * torch.sigmoid(logit)


def benefit_label(loss_null, loss_cond, eps=1e-8):
    """Soft target in (0, 1) for strictly positive losses. Gradient is stopped."""
    loss_null = loss_null.float()
    loss_cond = loss_cond.float()
    label = loss_null / (loss_null + loss_cond + float(eps))
    return label.detach()


def masked_cosine(query, key, eps=1e-8):
    """Row-wise cosine. Returns a detached tensor so the gate cannot train the denoiser."""
    query = query.float()
    key = key.float()
    similarity = F.cosine_similarity(query, key, dim=-1, eps=eps)
    return similarity.detach()
