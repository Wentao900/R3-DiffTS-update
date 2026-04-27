import torch
import torch.nn as nn


REGIME_NAMES = ('up', 'down', 'flat', 'turning', 'volatile')


class ForecastPolicyController(nn.Module):
    def __init__(self, pred_len, config=None):
        super().__init__()
        cfg = config or {}
        self.pred_len = int(pred_len)
        self.enabled = bool(cfg.get('enabled', False))
        self.fine_topk_ratio = float(cfg.get('fine_topk_ratio', 0.35))
        self.max_fine_ratio = float(cfg.get('max_fine_ratio', 0.6))
        self.min_fine_points = int(cfg.get('min_fine_points', 1))
        self.rag_invalid_scale = float(cfg.get('rag_invalid_scale', 0.25))
        self.unclear_trend_scale = float(cfg.get('unclear_trend_scale', 0.5))
        self.sample_budget_floor = float(cfg.get('sample_budget_floor', 0.5))
        self.sample_budget_ceiling = float(cfg.get('sample_budget_ceiling', 1.0))
        self.turning_center = float(cfg.get('turning_center', 0.35))
        self.turning_width = float(cfg.get('turning_width', 0.18))

    def _history_stats(self, history):
        series = history.mean(dim=1)
        history_len = max(series.shape[-1], 2)
        slope = (series[:, -1] - series[:, 0]) / float(history_len - 1)
        split = max(history_len // 2, 1)
        early = series[:, : split + 1]
        late = series[:, split:]
        early_slope = (early[:, -1] - early[:, 0]) / float(max(early.shape[-1] - 1, 1))
        late_slope = (late[:, -1] - late[:, 0]) / float(max(late.shape[-1] - 1, 1))
        diffs = series[:, 1:] - series[:, :-1]
        local_vol = diffs.std(dim=1)
        level = series.abs().mean(dim=1) + 1e-6
        flat_score = 1.0 - (slope.abs() / (local_vol + slope.abs() + 1e-6))
        turning_score = (early_slope * late_slope < 0).float()
        turning_score = torch.maximum(turning_score, (early_slope - late_slope).abs() / (local_vol + 1e-6))
        turning_score = turning_score.clamp(0.0, 1.0)
        volatile_score = (local_vol / level).clamp(0.0, 1.0)
        return {
            'slope': slope,
            'local_vol': local_vol,
            'flat_score': flat_score.clamp(0.0, 1.0),
            'turning_score': turning_score,
            'volatile_score': volatile_score,
            'tail_diffs': diffs[:, -min(diffs.shape[-1], 8):] if diffs.shape[-1] > 0 else diffs,
        }

    def _infer_regime(self, stats, trend_prior):
        direction = trend_prior[:, 0].clamp(-1.0, 1.0)
        volatility = trend_prior[:, 2].clamp(0.0, 1.0)
        up = torch.relu(direction) * (1.0 - stats['turning_score']) * (1.0 - stats['volatile_score'])
        down = torch.relu(-direction) * (1.0 - stats['turning_score']) * (1.0 - stats['volatile_score'])
        flat = (1.0 - direction.abs()) * stats['flat_score'] * (1.0 - stats['volatile_score'])
        turning = stats['turning_score'] * (1.0 - stats['volatile_score'] * 0.5)
        volatile = torch.maximum(volatility, stats['volatile_score'])
        regime_logits = torch.stack([up, down, flat, turning, volatile], dim=1) + 1e-4
        regime_probs = regime_logits / regime_logits.sum(dim=1, keepdim=True)
        regime_index = regime_probs.argmax(dim=1)
        regime_token = torch.nn.functional.one_hot(regime_index, num_classes=len(REGIME_NAMES)).float()
        return regime_probs, regime_index, regime_token

    def _build_horizon_risk(self, stats, evidence_conflict, uncertainty_score):
        device = evidence_conflict.device
        horizon = torch.linspace(0.0, 1.0, self.pred_len, device=device).unsqueeze(0)
        turning_curve = torch.exp(-0.5 * ((horizon - self.turning_center) / max(self.turning_width, 1e-3)) ** 2)
        late_curve = 0.3 + 0.7 * torch.sqrt(horizon + 1e-6)
        if stats['tail_diffs'].numel() == 0:
            local_curve = torch.ones((evidence_conflict.shape[0], self.pred_len), device=device)
        else:
            tail = stats['tail_diffs'].abs().unsqueeze(1)
            local_curve = torch.nn.functional.interpolate(tail, size=self.pred_len, mode='linear', align_corners=False).squeeze(1)
            local_curve = local_curve / (local_curve.amax(dim=1, keepdim=True) + 1e-6)
        risk = (
            0.35 * stats['turning_score'].unsqueeze(1) * turning_curve +
            0.25 * stats['volatile_score'].unsqueeze(1) * late_curve +
            0.20 * evidence_conflict.unsqueeze(1) +
            0.20 * uncertainty_score.unsqueeze(1)
        )
        risk = (risk + 0.25 * local_curve).clamp(0.0, 1.0)
        return risk

    def forward(self, history, trend_prior, text_mask=None, retrieval_mask=None, reasoning_mask=None):
        batch_size = history.shape[0]
        device = history.device
        stats = self._history_stats(history)
        numeric_direction = torch.sign(stats['slope'])
        prior_direction = torch.sign(trend_prior[:, 0])
        evidence_conflict = 0.5 * (numeric_direction - prior_direction).abs()
        volatility = trend_prior[:, 2].clamp(0.0, 1.0)
        strength = ((trend_prior[:, 1] - 0.5) / 1.0).clamp(0.0, 1.0)
        uncertainty_score = (0.45 * volatility + 0.35 * stats['turning_score'] + 0.20 * evidence_conflict).clamp(0.0, 1.0)
        clarity = (strength * (1.0 - volatility) * (1.0 - 0.5 * stats['turning_score']) * (1.0 - 0.5 * evidence_conflict)).clamp(0.0, 1.0)
        regime_probs, regime_index, regime_token = self._infer_regime(stats, trend_prior)

        if text_mask is None:
            text_mask = torch.ones((batch_size,), device=device)
        if retrieval_mask is None:
            retrieval_mask = text_mask.clone()
        if reasoning_mask is None:
            reasoning_mask = text_mask.clone()

        rag_gate = retrieval_mask.float() * (1.0 - evidence_conflict)
        if not self.enabled:
            rag_gate = retrieval_mask.float()
        rag_gate = torch.maximum(rag_gate, self.rag_invalid_scale * retrieval_mask.float())
        rag_gate = torch.minimum(rag_gate, torch.ones_like(rag_gate))

        cot_gate = reasoning_mask.float() * (0.5 + 0.5 * clarity)
        cot_gate = cot_gate.clamp(0.0, 1.0)
        trend_gate = clarity + (1.0 - clarity) * (1.0 - self.unclear_trend_scale)
        guidance_scale = (0.5 * rag_gate + 0.5 * cot_gate) * trend_gate
        guidance_scale = guidance_scale.clamp(0.05, 1.0)

        risk_scores = self._build_horizon_risk(stats, evidence_conflict, uncertainty_score)
        risk_mean = risk_scores.mean(dim=1)
        fine_ratio = (self.fine_topk_ratio * (0.5 + 0.5 * risk_mean)).clamp(0.05, self.max_fine_ratio)
        num_fine = torch.clamp((fine_ratio * self.pred_len).round().long(), min=self.min_fine_points, max=self.pred_len)
        fine_mask = torch.zeros((batch_size, self.pred_len), device=device)
        for idx in range(batch_size):
            top_idx = torch.topk(risk_scores[idx], k=int(num_fine[idx].item()), largest=True).indices
            fine_mask[idx, top_idx] = 1.0

        residual_scale = (1.0 + 0.30 * uncertainty_score + 0.20 * regime_probs[:, 4]).clamp(0.75, 1.75)
        sample_budget_scale = self.sample_budget_floor + (self.sample_budget_ceiling - self.sample_budget_floor) * risk_mean
        sample_budget_scale = sample_budget_scale.clamp(self.sample_budget_floor, self.sample_budget_ceiling)

        return {
            'regime_probs': regime_probs,
            'regime_index': regime_index,
            'regime_token': regime_token,
            'turning_score': stats['turning_score'],
            'volatile_score': stats['volatile_score'],
            'evidence_conflict': evidence_conflict,
            'uncertainty_score': uncertainty_score,
            'clarity_score': clarity,
            'rag_gate': rag_gate,
            'cot_gate': cot_gate,
            'trend_gate': trend_gate,
            'guidance_scale': guidance_scale,
            'residual_scale': residual_scale,
            'sample_budget_scale': sample_budget_scale,
            'fine_ratio': fine_ratio,
            'fine_mask': fine_mask,
            'risk_scores': risk_scores,
        }
