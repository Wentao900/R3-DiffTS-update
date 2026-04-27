import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastProjectionHead(nn.Module):
    def __init__(self, lookback_len, output_len, cond_dim=0, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.lookback_len = int(lookback_len)
        self.output_len = int(output_len)
        self.cond_dim = int(cond_dim)
        self.hist_proj = nn.Sequential(
            nn.Linear(self.lookback_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.cond_proj = nn.Linear(self.cond_dim, hidden_dim) if self.cond_dim > 0 else None
        self.output_proj = nn.Linear(hidden_dim, self.output_len)

    def forward(self, history, cond_vec=None):
        batch_size, feature_dim, _ = history.shape
        hidden = self.hist_proj(history.reshape(batch_size * feature_dim, self.lookback_len))
        if self.cond_proj is not None and cond_vec is not None:
            cond_hidden = self.cond_proj(cond_vec).unsqueeze(1).expand(-1, feature_dim, -1)
            hidden = hidden + cond_hidden.reshape(batch_size * feature_dim, -1)
        forecast = self.output_proj(hidden).reshape(batch_size, feature_dim, self.output_len)
        return forecast


class CoarseForecastHead(nn.Module):
    def __init__(self, lookback_len, pred_len, cond_dim=0, hidden_dim=128, coarse_factor=4, dropout=0.0):
        super().__init__()
        self.pred_len = int(pred_len)
        self.coarse_factor = max(int(coarse_factor), 1)
        self.coarse_len = max(int(math.ceil(self.pred_len / self.coarse_factor)), 1)
        self.core = ForecastProjectionHead(
            lookback_len=lookback_len,
            output_len=self.coarse_len,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, history, cond_vec=None):
        coarse = self.core(history, cond_vec=cond_vec)
        batch_size, feature_dim, coarse_len = coarse.shape
        upsampled = F.interpolate(
            coarse.reshape(batch_size * feature_dim, 1, coarse_len),
            size=self.pred_len,
            mode='linear',
            align_corners=False,
        )
        return upsampled.reshape(batch_size, feature_dim, self.pred_len)


class UncertaintyHead(nn.Module):
    def __init__(self, lookback_len, pred_len, cond_dim=0, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.loc_head = ForecastProjectionHead(
            lookback_len=lookback_len,
            output_len=pred_len,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.logvar_head = ForecastProjectionHead(
            lookback_len=lookback_len,
            output_len=pred_len,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, history, cond_vec=None):
        loc = self.loc_head(history, cond_vec=cond_vec)
        logvar = self.logvar_head(history, cond_vec=cond_vec).clamp(min=-6.0, max=4.0)
        return loc, logvar


def mean_pool_context(context):
    if context is None:
        return None
    if context.dim() != 3:
        raise ValueError(f'context must have shape (B, C, T), got {tuple(context.shape)}')
    return context.mean(dim=-1)
