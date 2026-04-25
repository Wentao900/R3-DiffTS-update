import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from diff_models import diff_CSDI
from utils.prepare4llm import get_llm

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=25):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class CSDI_series_decomp(nn.Module):
    def __init__(self, lookback_len, pred_len, kernel_size=25):
        super(CSDI_series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.lookback_len = lookback_len
        self.pred_len = pred_len

    def forward(self, x):
        x = x.permute(0, 2, 1)
        lookback = x[:, :self.lookback_len, :]

        moving_mean = self.moving_avg(lookback)
        res = lookback - moving_mean
        
        moving_mean = moving_mean.permute(0, 2, 1)
        res = res.permute(0, 2, 1)

        moving_mean = nn.functional.pad(moving_mean, (0, self.pred_len), "constant", 0)
        res = nn.functional.pad(res, (0, self.pred_len), "constant", 0)
        return res, moving_mean
        

    

class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device, window_lens):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.timestep_branch = config["model"]["timestep_branch"]
        self.timestep_emb_cat = config["model"]["timestep_emb_cat"]
        self.with_texts = config["model"]["with_texts"]
        self.noise_esti = config["diffusion"]["noise_esti"]
        self.relative_size_emb_cat = config["model"]["relative_size_emb_cat"]
        self.decomp = config["model"]["decomp"]
        self.ddim = config["diffusion"]["ddim"]
        self.sample_steps = config["diffusion"]["sample_steps"]
        self.sample_method = config["diffusion"]["sample_method"]
        train_cfg = config.get("train", {})

        self.lookback_len = config["model"]["lookback_len"]
        self.pred_len = config["model"]["pred_len"]
        self.diff_channels = config["diffusion"]["channels"]
        self.cfg = config["diffusion"]["cfg"]
        self.trend_cfg = config["diffusion"].get("trend_cfg", False)
        self.trend_cfg_power = config["diffusion"].get("trend_cfg_power", 1.0)
        self.trend_cfg_random = config["diffusion"].get("trend_cfg_random", False)
        self.trend_strength_scale = config["diffusion"].get("trend_strength_scale", 1.0)
        self.trend_volatility_scale = config["diffusion"].get("trend_volatility_scale", 1.0)
        self.trend_time_floor = config["diffusion"].get("trend_time_floor", 0.0)
        self.c_mask_prob = config["diffusion"]["c_mask_prob"]
        self.context_dim = config["model"]["context_dim"]
        self.llm = config["model"]["llm"]
        self.domain = config["model"]["domain"]
        self.save_attn = config["model"]["save_attn"]
        self.save_token = config["model"]["save_token"]
        self.text_max_length = int(config["model"].get("text_max_length", 192))
        self.text_encode_batch_size = int(config["model"].get("text_encode_batch_size", 8))

        self.multi_res_horizons = train_cfg.get("multi_res_horizons", [])
        self.multi_res_loss_weight = float(train_cfg.get("multi_res_loss_weight", 0.0))
        self.multi_res_use_huber = bool(train_cfg.get("multi_res_use_huber", True))
        self.multi_res_huber_delta = float(train_cfg.get("multi_res_huber_delta", 1.0))
        self.multi_res_huber_deltas_cfg = train_cfg.get("multi_res_huber_deltas", None)
        self.multi_res_huber_delta_mode = str(train_cfg.get("multi_res_huber_delta_mode", "fallback_uniform"))
        self.multi_res_huber_delta_scale = float(train_cfg.get("multi_res_huber_delta_scale", 0.5))
        self.multi_res_dynamic = bool(train_cfg.get("multi_res_dynamic", False))
        self.multi_res_dynamic_by_t = bool(train_cfg.get("multi_res_dynamic_by_t", True))
        self.multi_res_dynamic_by_epoch = bool(train_cfg.get("multi_res_dynamic_by_epoch", True))
        self.multi_res_dynamic_by_trend = bool(train_cfg.get("multi_res_dynamic_by_trend", True))
        self.multi_res_dynamic_min_weight = float(train_cfg.get("multi_res_dynamic_min_weight", 0.2))
        self.multi_res_progressive = bool(train_cfg.get("multi_res_progressive", False))
        self.multi_res_ema_alpha = float(train_cfg.get("multi_res_ema_alpha", 0.05))
        self.multi_res_difficulty_weight = float(train_cfg.get("multi_res_difficulty_weight", 0.0))
        self.multi_res_group_balance = bool(train_cfg.get("multi_res_group_balance", True))
        self.multi_res_group_max_ratio = float(train_cfg.get("multi_res_group_max_ratio", 2.0))
        self.multi_res_segment_loss = bool(train_cfg.get("multi_res_segment_loss", True))
        self.multi_res_reliability_weight = float(train_cfg.get("multi_res_reliability_weight", 1.0))
        self.multi_res_difficulty_inverse = bool(train_cfg.get("multi_res_difficulty_inverse", True))
        self.multi_res_difficulty_gamma = float(train_cfg.get("multi_res_difficulty_gamma", 0.5))
        self.auxiliary_loss_max_ratio = float(train_cfg.get("auxiliary_loss_max_ratio", 0.0))
        self.current_epoch = 0
        self.total_epochs = max(int(train_cfg.get("epochs", 1)), 1)
        self.pattern_residual_diffusion = bool(config["model"].get("pattern_residual_diffusion", True))
        self.pattern_text_evidence = bool(config["model"].get("pattern_text_evidence", True))
        self.pattern_hidden_dim = int(config["model"].get("pattern_hidden_dim", 64))
        self.pattern_stats_dim = 8
        self.pattern_num_experts = 5
        self.pattern_consistency_weight = float(train_cfg.get("pattern_consistency_weight", 0.03))
        self.pattern_expert_weight = float(train_cfg.get("pattern_expert_weight", 0.05))
        self.pattern_baseline_detach = bool(train_cfg.get("pattern_baseline_detach", False))
        self.pattern_router_temperature = float(config["model"].get("pattern_router_temperature", 1.0))
        self.guide_w_default = float(config["model"].get("guide_w_default", 1.0))
        self.pattern_reliability = bool(config["model"].get("pattern_reliability", True))
        self.pattern_reliability_threshold = float(config["model"].get("pattern_reliability_threshold", 0.35))
        self.pattern_reliability_temperature = float(config["model"].get("pattern_reliability_temperature", 0.1))
        self.pattern_reliability_min = float(config["model"].get("pattern_reliability_min", 0.05))
        self.pattern_reliability_max = float(config["model"].get("pattern_reliability_max", 1.0))
        self.pattern_aux_reliability = bool(config["model"].get("pattern_aux_reliability", True))
        self.pattern_text_drop_prob = float(config["model"].get("pattern_text_drop_prob", 0.0))
        self.multi_res_horizons = self._sanitize_multi_res_horizons(self.multi_res_horizons)
        self.multi_res_horizon_to_index = {
            int(horizon): idx for idx, horizon in enumerate(self.multi_res_horizons)
        }
        self.multi_res_horizon_reliabilities = self._resolve_multi_res_horizon_reliabilities(
            train_cfg.get("multi_res_horizon_reliabilities", None)
        )
        difficulty_size = max(len(self.multi_res_horizons), 1)
        self.register_buffer(
            "multi_res_difficulty_ema",
            torch.ones(difficulty_size, dtype=torch.float32),
        )
        reliability_size = max(len(self.multi_res_horizons), 1)
        reliability_tensor = torch.ones(reliability_size, dtype=torch.float32)
        if len(self.multi_res_horizon_reliabilities) > 0:
            reliability_tensor = torch.tensor(self.multi_res_horizon_reliabilities, dtype=torch.float32)
        self.register_buffer("multi_res_reliability", reliability_tensor)
        self.multi_res_huber_deltas = self._resolve_multi_res_huber_deltas(
            self.multi_res_horizons, self.multi_res_huber_deltas_cfg
        )

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1 
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
            
        if self.decomp:
            self.decomposition = CSDI_series_decomp(self.lookback_len, self.pred_len, kernel_size=25)

        if self.timestep_emb_cat:
            self.timestep_emb = nn.Sequential(nn.Linear(config["model"]["timestep_dim"], self.diff_channels//8), 
                                      nn.LayerNorm(self.diff_channels//8),
                                      nn.ReLU(),
                                      nn.Linear(self.diff_channels//8, self.diff_channels//4), 
                                      nn.LayerNorm(self.diff_channels//4),
                                      nn.ReLU())
        if self.timestep_branch:
            # Predict series directly from timestep features for TTF branch
            self.timestep_pred = nn.Sequential(
                nn.Conv1d(config["model"]["timestep_dim"], self.diff_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(self.diff_channels, self.target_dim, kernel_size=1),
            )
        
        if self.relative_size_emb_cat:
            self.relative_size_emb = nn.Sequential(nn.Linear(self.lookback_len, self.lookback_len), 
                                                   nn.LayerNorm(self.lookback_len),
                                                   nn.ReLU(),
                                                   nn.Linear(self.lookback_len, self.diff_channels),
                                                   nn.LayerNorm(self.diff_channels),
                                                   nn.ReLU(),)

        if self.with_texts:
            self.text_encoder, self.tokenizer = get_llm(self.llm, config["model"]["llm_layers"])
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            if self.llm != 'bert':
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    pad_token = '[PAD]'
                    self.tokenizer.add_special_tokens({'pad_token': pad_token})
                    self.tokenizer.pad_token = pad_token
            text_hidden_dim = getattr(getattr(self.text_encoder, "config", None), "hidden_size", self.context_dim)
            self.text_hidden_dim = int(text_hidden_dim)

        pattern_input_dim = self.pattern_stats_dim + self.pattern_num_experts
        self.pattern_router = nn.Sequential(
            nn.Linear(pattern_input_dim, self.pattern_hidden_dim),
            nn.LayerNorm(self.pattern_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.pattern_hidden_dim, self.pattern_num_experts),
        )
        self.pattern_evidence_head = nn.Sequential(
            nn.Linear(7, self.pattern_hidden_dim),
            nn.LayerNorm(self.pattern_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.pattern_hidden_dim, self.pattern_num_experts),
        )
        if self.with_texts:
            self.pattern_text_head = nn.Sequential(
                nn.Linear(self.text_hidden_dim, self.pattern_hidden_dim),
                nn.LayerNorm(self.pattern_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.pattern_hidden_dim, self.pattern_num_experts),
            )
        else:
            self.pattern_text_head = None

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["decomp"] = self.decomp
        config_diff["lookback_len"] = self.lookback_len
        config_diff["pred_len"] = self.pred_len
        config_diff["with_timestep"] = True if self.timestep_emb_cat else False
        config_diff["context_dim"] = self.context_dim
        config_diff["with_texts"] = self.with_texts
        config_diff["time_weight"] = config["diffusion"]["time_weight"]
        config_diff["save_attn"] = config["model"]["save_attn"]

        input_dim = 1 if self.is_unconditional == True else 2
        mode_num = 1

        if self.decomp:
            self.diffmodel_trend = diff_CSDI(config_diff, input_dim, mode_num=mode_num)
            self.diffmodel_sesonal = diff_CSDI(config_diff, input_dim, mode_num=mode_num)
        else:
            self.diffmodel = diff_CSDI(config_diff, input_dim, mode_num=mode_num)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def _sanitize_multi_res_horizons(self, horizons):
        if isinstance(horizons, int):
            horizons = [horizons]
        elif horizons is None:
            horizons = []
        sanitized = []
        for horizon in horizons:
            try:
                horizon = int(horizon)
            except (TypeError, ValueError):
                continue
            if 1 <= horizon <= int(self.pred_len):
                sanitized.append(horizon)

        if len(sanitized) == 0 and self.multi_res_loss_weight > 0 and self.pred_len > 0:
            if self.pred_len <= 4:
                sanitized = list(range(1, self.pred_len + 1))
            else:
                sanitized = [
                    1,
                    int(math.ceil(self.pred_len / 4.0)),
                    int(math.ceil(self.pred_len / 2.0)),
                    int(self.pred_len),
                ]

        return sorted(set(sanitized))

    def _resolve_multi_res_horizon_reliabilities(self, reliabilities):
        if len(self.multi_res_horizons) == 0:
            return []
        if reliabilities is None:
            return [1.0 for _ in self.multi_res_horizons]
        if isinstance(reliabilities, (int, float)):
            values = [float(reliabilities)]
        elif isinstance(reliabilities, (list, tuple)):
            values = []
            for value in reliabilities:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    values.append(1.0)
        else:
            values = [1.0 for _ in self.multi_res_horizons]
        if len(values) != len(self.multi_res_horizons):
            warnings.warn(
                "multi_res_horizon_reliabilities length does not match final multi_res_horizons; falling back to 1.0.",
                RuntimeWarning,
            )
            values = [1.0 for _ in self.multi_res_horizons]
        return [float(min(max(value, 0.0), 1.0)) for value in values]

    def _resolve_multi_res_huber_deltas(self, horizons, delta_cfg):
        if len(horizons) == 0:
            return []

        fallback = [float(max(self.multi_res_huber_delta, 1e-6))] * len(horizons)
        if self.multi_res_huber_delta_mode == "functional":
            return [
                float(
                    max(
                        self.multi_res_huber_delta
                        * (1.0 + self.multi_res_huber_delta_scale * (float(horizon) / max(float(self.pred_len), 1.0))),
                        1e-6,
                    )
                )
                for horizon in horizons
            ]
        if delta_cfg is None:
            return fallback
        if isinstance(delta_cfg, (int, float)):
            return [float(max(delta_cfg, 1e-6))] * len(horizons)
        if not isinstance(delta_cfg, (list, tuple)):
            warnings.warn(
                "multi_res_huber_deltas is not a list/tuple; falling back to uniform delta.",
                RuntimeWarning,
            )
            return fallback
        if len(delta_cfg) != len(horizons):
            warnings.warn(
                "multi_res_huber_deltas length does not match final multi_res_horizons; falling back to uniform delta.",
                RuntimeWarning,
            )
            return fallback

        resolved = []
        try:
            for value in delta_cfg:
                resolved.append(float(max(value, 1e-6)))
        except (TypeError, ValueError):
            warnings.warn(
                "multi_res_huber_deltas contains invalid values; falling back to uniform delta.",
                RuntimeWarning,
            )
            return fallback
        return resolved

    def _get_active_multi_res_horizons(self):
        horizons = list(self.multi_res_horizons)
        if len(horizons) <= 1 or not self.multi_res_progressive:
            return horizons
        if self.total_epochs <= 1:
            return horizons

        progress = float(min(max(self.current_epoch, 0), self.total_epochs - 1)) / float(self.total_epochs - 1)
        active_count = 1 + int(math.floor(progress * (len(horizons) - 1) + 1e-8))
        active_count = min(max(active_count, 1), len(horizons))
        return horizons[:active_count]

    def _get_horizon_group(self, horizon):
        short_end = max(1, int(math.ceil(self.pred_len / 4.0)))
        mid_end = max(short_end + 1, int(math.ceil(self.pred_len / 2.0)))
        if int(horizon) <= short_end:
            return "short"
        if int(horizon) <= mid_end:
            return "mid"
        return "long"

    def get_multi_res_debug_state(self):
        active_horizons = self._get_active_multi_res_horizons()
        indices = [self.multi_res_horizon_to_index[h] for h in active_horizons if h in self.multi_res_horizon_to_index]
        difficulty = []
        if indices:
            difficulty = self.multi_res_difficulty_ema[indices].detach().cpu().tolist()
        return {
            "final_horizons": list(self.multi_res_horizons),
            "active_horizons": list(active_horizons),
            "huber_deltas": list(self.multi_res_huber_deltas),
            "horizon_reliabilities": list(self.multi_res_horizon_reliabilities),
            "difficulty_ema": difficulty,
            "horizon_groups": [self._get_horizon_group(horizon) for horizon in active_horizons],
            "segment_loss": self.multi_res_segment_loss,
            "difficulty_inverse": self.multi_res_difficulty_inverse,
        }

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else: 
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask


    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim) 
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K, emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1) 
        side_info = side_info.permute(0, 3, 2, 1) 

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1) 
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def _masked_mean(self, values, mask, dim, keepdim=True):
        denom = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1.0)
        return (values * mask).sum(dim=dim, keepdim=keepdim) / denom

    def _safe_lag_corr(self, series, mask, lag):
        if lag <= 0 or series.shape[1] <= lag:
            return torch.zeros((series.shape[0],), device=series.device, dtype=series.dtype)
        x0 = series[:, :-lag]
        x1 = series[:, lag:]
        m = mask[:, :-lag] * mask[:, lag:]
        denom = m.sum(dim=1).clamp(min=1.0)
        mu0 = (x0 * m).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
        mu1 = (x1 * m).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
        z0 = (x0 - mu0) * m
        z1 = (x1 - mu1) * m
        cov = (z0 * z1).sum(dim=1)
        var0 = (z0 ** 2).sum(dim=1)
        var1 = (z1 ** 2).sum(dim=1)
        return cov / torch.sqrt((var0 * var1).clamp(min=1e-6))

    def _compute_pattern_stats(self, observed_data, cond_mask):
        B, _, L = observed_data.shape
        hist_len = min(max(int(self.lookback_len), 1), L)
        hist = observed_data[:, :, :hist_len]
        hist_mask = cond_mask[:, :, :hist_len].float()
        time_mask = (hist_mask.sum(dim=1) > 0).float()
        series = (hist * hist_mask).sum(dim=1) / hist_mask.sum(dim=1).clamp(min=1.0)
        series_mean = self._masked_mean(series, time_mask, dim=1, keepdim=True)
        series = torch.where(time_mask > 0, series, series_mean.expand_as(series))
        t = torch.linspace(-1.0, 1.0, hist_len, device=observed_data.device, dtype=observed_data.dtype).unsqueeze(0)
        t_mean = self._masked_mean(t.expand(B, -1), time_mask, dim=1, keepdim=True)
        x_mean = self._masked_mean(series, time_mask, dim=1, keepdim=True)
        t_center = (t - t_mean) * time_mask
        x_center = (series - x_mean) * time_mask
        slope = (t_center * x_center).sum(dim=1) / (t_center ** 2).sum(dim=1).clamp(min=1e-6)
        fitted = x_mean + slope.unsqueeze(1) * (t - t_mean)
        ss_res = (((series - fitted) * time_mask) ** 2).sum(dim=1)
        ss_tot = (((series - x_mean) * time_mask) ** 2).sum(dim=1).clamp(min=1e-6)
        trend_r2 = (1.0 - ss_res / ss_tot).clamp(min=0.0, max=1.0)
        acf1 = self._safe_lag_corr(series, time_mask, 1).clamp(min=-1.0, max=1.0)
        period_candidates = [2, 3, 4, 6, 7, 12, 24, 52, 96]
        season_scores = [
            self._safe_lag_corr(series, time_mask, p).clamp(min=0.0, max=1.0)
            for p in period_candidates
            if hist_len > p
        ]
        if len(season_scores) > 0:
            season_strength = torch.stack(season_scores, dim=1).max(dim=1).values
        else:
            season_strength = torch.zeros((B,), device=observed_data.device, dtype=observed_data.dtype)
        diff = series[:, 1:] - series[:, :-1] if hist_len > 1 else torch.zeros((B, 1), device=observed_data.device, dtype=observed_data.dtype)
        diff_mask = time_mask[:, 1:] * time_mask[:, :-1] if hist_len > 1 else torch.zeros_like(diff)
        diff_abs = diff.abs() * diff_mask
        diff_mean = diff_abs.sum(dim=1) / diff_mask.sum(dim=1).clamp(min=1.0)
        diff_std = torch.sqrt((((diff_abs - diff_mean.unsqueeze(1)) * diff_mask) ** 2).sum(dim=1) / diff_mask.sum(dim=1).clamp(min=1.0) + 1e-6)
        volatility = diff_std / (series.std(dim=1, unbiased=False).clamp(min=1e-3))
        event_score = torch.sigmoid(diff_abs.max(dim=1).values / (diff_mean + diff_std + 1e-3) - 2.0)
        state_score = (acf1.clamp(min=0.0, max=1.0) * torch.exp(-volatility.clamp(min=0.0, max=5.0))).clamp(min=0.0, max=1.0)
        last = series[:, -1]
        mean_revert = torch.sigmoid((last - x_mean.squeeze(1)).abs() / series.std(dim=1, unbiased=False).clamp(min=1e-3) - 1.0)
        trend_strength = trend_r2 * torch.tanh(slope.abs())
        stats = torch.stack(
            [
                trend_strength,
                trend_r2,
                acf1.clamp(min=0.0, max=1.0),
                season_strength,
                state_score,
                mean_revert,
                event_score,
                volatility.clamp(min=0.0, max=5.0) / 5.0,
            ],
            dim=1,
        )
        return stats.clamp(min=0.0, max=1.0)

    def _compute_pattern_text_evidence(self, text_pooled, text_mask, text_evidence_vec, batch_size, device):
        if not self.pattern_text_evidence:
            return torch.zeros((batch_size, self.pattern_num_experts), device=device)
        logits = torch.zeros((batch_size, self.pattern_num_experts), device=device)
        has_signal = False
        if text_evidence_vec is not None:
            evidence = text_evidence_vec.float()
            if evidence.dim() == 1:
                evidence = evidence.reshape(batch_size, -1)
            if evidence.shape[1] < 7:
                evidence = F.pad(evidence, (0, 7 - evidence.shape[1]))
            logits = logits + self.pattern_evidence_head(evidence[:, :7])
            has_signal = True
        if self.with_texts and self.pattern_text_head is not None and text_pooled is not None:
            logits = logits + self.pattern_text_head(text_pooled.float())
            has_signal = True
        if not has_signal:
            return torch.zeros_like(logits)
        evidence = F.softmax(logits, dim=-1)
        if text_mask is not None:
            availability = text_mask.float()
            if availability.dim() > 1:
                availability = availability.mean(dim=1)
            evidence = evidence * (availability.reshape(-1, 1) > 0).float()
        return evidence

    def _maybe_drop_pattern_text(self, text_pattern):
        if (not self.training) or self.pattern_text_drop_prob <= 0:
            return text_pattern
        keep_prob = 1.0 - min(max(float(self.pattern_text_drop_prob), 0.0), 1.0)
        keep = torch.bernoulli(torch.full((text_pattern.shape[0], 1), keep_prob, device=text_pattern.device))
        return text_pattern * keep

    def _compute_pattern_reliability(self, stats, text_pattern):
        if not self.pattern_reliability:
            return torch.ones((stats.shape[0],), device=stats.device, dtype=stats.dtype)
        trend_r2 = stats[:, 1]
        season_strength = stats[:, 3]
        state_score = stats[:, 4]
        stability = 1.0 - stats[:, 7].clamp(min=0.0, max=1.0)
        text_strength = text_pattern.max(dim=1).values if text_pattern.numel() > 0 else torch.zeros_like(trend_r2)
        score = (
            0.35 * season_strength
            + 0.25 * trend_r2
            + 0.20 * state_score
            + 0.10 * stability
            + 0.10 * text_strength
        )
        temperature = max(float(self.pattern_reliability_temperature), 1e-6)
        rho = torch.sigmoid((score - float(self.pattern_reliability_threshold)) / temperature)
        rho_min = min(max(float(self.pattern_reliability_min), 0.0), 1.0)
        rho_max = min(max(float(self.pattern_reliability_max), rho_min), 1.0)
        return (rho_min + (rho_max - rho_min) * rho).clamp(min=rho_min, max=rho_max)

    def _build_pattern_baselines(self, observed_data, cond_mask, stats):
        B, K, L = observed_data.shape
        baseline = torch.zeros_like(observed_data)
        future_len = min(max(int(self.pred_len), 0), max(L - self.lookback_len, 0))
        if future_len <= 0:
            experts = torch.zeros((B, self.pattern_num_experts, K, 0), device=observed_data.device, dtype=observed_data.dtype)
            return baseline, experts
        hist_len = min(max(int(self.lookback_len), 1), L)
        hist = observed_data[:, :, :hist_len]
        hist_mask = cond_mask[:, :, :hist_len].float()
        mean = self._masked_mean(hist, hist_mask, dim=2, keepdim=True)
        filled_hist = torch.where(hist_mask > 0, hist, mean.expand_as(hist))
        first = filled_hist[:, :, 0:1]
        last = filled_hist[:, :, -1:]
        prev = filled_hist[:, :, -2:-1] if hist_len > 1 else last
        slope = (last - first) / max(hist_len - 1, 1)
        steps = torch.arange(1, future_len + 1, device=observed_data.device, dtype=observed_data.dtype).view(1, 1, -1)
        trend_future = last + slope * steps
        state_future = last.expand(-1, -1, future_len)
        phi = torch.exp(-steps / max(float(future_len), 1.0))
        mean_future = mean + phi * (last - mean)
        event_future = last + (last - prev) * phi

        season_candidates = []
        season_weights = []
        for p in [2, 3, 4, 6, 7, 12, 24, 52, 96]:
            if hist_len <= p:
                continue
            idx = hist_len - p + (torch.arange(future_len, device=observed_data.device) % p)
            season_candidates.append(filled_hist[:, :, idx])
            series = filled_hist.mean(dim=1)
            time_mask = (hist_mask.sum(dim=1) > 0).float()
            season_weights.append(self._safe_lag_corr(series, time_mask, p).clamp(min=0.0, max=1.0))
        if len(season_candidates) > 0:
            candidate_tensor = torch.stack(season_candidates, dim=1)
            weight_tensor = torch.stack(season_weights, dim=1)
            weight_tensor = F.softmax(4.0 * weight_tensor, dim=1).view(B, -1, 1, 1)
            season_future = (candidate_tensor * weight_tensor).sum(dim=1)
        else:
            season_future = state_future

        expert_futures = torch.stack(
            [trend_future, season_future, state_future, mean_future, event_future],
            dim=1,
        )
        return baseline, expert_futures

    def _compute_pattern_outputs(self, observed_data, cond_mask, text_pooled=None, text_mask=None, text_evidence_vec=None, guidance_scale=1.0):
        B, K, L = observed_data.shape
        stats = self._compute_pattern_stats(observed_data, cond_mask)
        text_pattern = self._compute_pattern_text_evidence(text_pooled, text_mask, text_evidence_vec, B, observed_data.device)
        text_pattern = self._maybe_drop_pattern_text(text_pattern)
        temperature = max(float(self.pattern_router_temperature), 1e-3)
        numeric_router_input = torch.cat([stats, torch.zeros_like(text_pattern)], dim=1)
        text_router_input = torch.cat([stats, text_pattern], dim=1)
        numeric_logits = self.pattern_router(numeric_router_input) / temperature
        text_logits = self.pattern_router(text_router_input) / temperature
        scale = float(guidance_scale)
        guided_logits = numeric_logits + scale * (text_logits - numeric_logits)
        pi = F.softmax(guided_logits, dim=-1)
        baseline, expert_futures = self._build_pattern_baselines(observed_data, cond_mask, stats)
        future_len = expert_futures.shape[-1]
        reliability = self._compute_pattern_reliability(stats, text_pattern)
        if future_len > 0:
            mixed_future = (pi.view(B, self.pattern_num_experts, 1, 1) * expert_futures).sum(dim=1)
            safe_future = expert_futures[:, 2]
            mixed_future = reliability.view(B, 1, 1) * mixed_future + (1.0 - reliability).view(B, 1, 1) * safe_future
            baseline[:, :, self.lookback_len:self.lookback_len + future_len] = mixed_future
        pseudo = torch.stack(
            [stats[:, 1], stats[:, 3], stats[:, 4], stats[:, 5], stats[:, 6]],
            dim=1,
        ).clamp(min=1e-4)
        pseudo = pseudo / pseudo.sum(dim=1, keepdim=True).clamp(min=1e-4)
        if self.pattern_baseline_detach:
            baseline = baseline.detach()
        return {
            "stats": stats,
            "text_pattern": text_pattern,
            "pi_numeric": F.softmax(numeric_logits, dim=-1),
            "pi_text": F.softmax(text_logits, dim=-1),
            "pi": pi,
            "pseudo": pseudo.detach(),
            "baseline": baseline,
            "expert_futures": expert_futures,
            "baseline_reliability": reliability,
            "guidance_scale": scale,
        }

    def _calc_pattern_aux_loss(self, pattern, observed_data, target_mask):
        aux = torch.zeros((), device=observed_data.device)
        if pattern is None:
            return aux
        pi = pattern["pi"].clamp(min=1e-6)
        pseudo = pattern["pseudo"].clamp(min=1e-6)
        reliability = pattern.get("baseline_reliability")
        if reliability is None:
            reliability = torch.ones((pi.shape[0],), device=observed_data.device, dtype=observed_data.dtype)
        reliability = reliability.reshape(-1).clamp(min=0.0, max=1.0)
        if not self.pattern_aux_reliability:
            reliability = torch.ones_like(reliability)
        if self.pattern_consistency_weight > 0:
            consistency = (pseudo * (pseudo.log() - pi.log())).sum(dim=1)
            consistency = (reliability * consistency).mean()
            aux = aux + self.pattern_consistency_weight * consistency
        if self.pattern_expert_weight > 0 and pattern["expert_futures"].numel() > 0:
            future_len = pattern["expert_futures"].shape[-1]
            start = int(self.lookback_len)
            end = start + future_len
            future_mask = target_mask[:, :, start:end].unsqueeze(1)
            future_target = observed_data[:, :, start:end].unsqueeze(1)
            residual = (pattern["expert_futures"] - future_target) * future_mask
            abs_res = residual.abs()
            huber = torch.where(abs_res <= 1.0, 0.5 * residual ** 2, abs_res - 0.5)
            denom = future_mask.sum(dim=(2, 3)).clamp(min=1.0)
            expert_loss = huber.sum(dim=(2, 3)) / denom
            weighted_expert_loss = (pattern["pi"] * expert_loss).sum(dim=1)
            weighted_expert_loss = (reliability * weighted_expert_loss).mean()
            aux = aux + self.pattern_expert_weight * weighted_expert_loss
        return aux

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, timesteps=None, timestep_emb=None, size_emb=None, context=None, trend_prior=None, pattern_text_pooled=None, pattern_text_mask=None, pattern_text_evidence_vec=None
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=context, trend_prior=trend_prior, pattern_text_pooled=pattern_text_pooled, pattern_text_mask=pattern_text_mask, pattern_text_evidence_vec=pattern_text_evidence_vec
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, timesteps=None, timestep_emb=None, size_emb=None, context=None, trend_prior=None, pattern_text_pooled=None, pattern_text_mask=None, pattern_text_evidence_vec=None, set_t=-1
    ):

        B, K, L = observed_data.shape
        if not self.noise_esti:
            means = torch.sum(observed_data*cond_mask, dim=2, keepdim=True) / torch.sum(cond_mask, dim=2, keepdim=True)
            stdev = torch.sqrt(torch.sum((observed_data - means) ** 2 * cond_mask, dim=2, keepdim=True) / (torch.sum(cond_mask, dim=2, keepdim=True) - 1) + 1e-5)
            observed_data = (observed_data - means) / stdev

        pattern = None
        pattern_baseline = None
        diffusion_target = observed_data
        if self.pattern_residual_diffusion and not self.noise_esti:
            pattern = self._compute_pattern_outputs(
                observed_data,
                cond_mask,
                text_pooled=pattern_text_pooled,
                text_mask=pattern_text_mask,
                text_evidence_vec=pattern_text_evidence_vec,
                guidance_scale=1.0,
            )
            pattern_baseline = pattern["baseline"]
            diffusion_target = observed_data - pattern_baseline

        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(diffusion_target)
        noisy_data = (current_alpha ** 0.5) * diffusion_target + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, diffusion_target, cond_mask)

        if self.cfg:
            cfg_mask = torch.bernoulli(torch.ones((B, )) - self.c_mask_prob).to(self.device)
        else:
            cfg_mask = None

        predicted = self._run_diffusion_model(
            total_input,
            side_info,
            t,
            cfg_mask,
            timestep_emb=timestep_emb,
            size_emb=size_emb,
            context=context,
        )

        if self.timestep_branch and timesteps is not None:
            predicted_from_timestep = self.timestep_pred(timesteps)
            predicted = 0.9 * predicted + 0.1 * predicted_from_timestep

        target_mask = observed_mask - cond_mask
        if self.noise_esti:
            residual = (noise - predicted) * target_mask 
        else:
            residual = (diffusion_target - predicted) * target_mask
        num_eval = target_mask.sum()
        main_loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        auxiliary_loss = torch.zeros((), device=observed_data.device)
        predicted_series = predicted
        if pattern_baseline is not None:
            predicted_series = predicted + pattern_baseline
            auxiliary_loss = auxiliary_loss + self._calc_pattern_aux_loss(pattern, observed_data, target_mask)
        if (not self.noise_esti) and self.multi_res_loss_weight > 0 and len(self.multi_res_horizons) > 0:
            aux_loss = self._calc_multi_res_loss(observed_data, predicted_series, target_mask, t=t, trend_prior=trend_prior)
            auxiliary_loss = auxiliary_loss + self.multi_res_loss_weight * aux_loss
        if self.auxiliary_loss_max_ratio > 0:
            aux_cap = max(self.auxiliary_loss_max_ratio, 0.0) * main_loss.detach()
            auxiliary_loss = torch.minimum(auxiliary_loss, aux_cap)
        return main_loss + auxiliary_loss

    def _unwrap_diffmodel_output(self, output):
        if isinstance(output, tuple):
            return output[0]
        return output

    def _run_diffusion_model(self, total_input, side_info, t, cfg_mask, timestep_emb=None, size_emb=None, context=None):
        if self.decomp:
            predicted_seasonal = self._unwrap_diffmodel_output(
                self.diffmodel_sesonal(total_input[0], side_info, t, cfg_mask, timestep_emb, size_emb, context)
            )
            predicted_trend = self._unwrap_diffmodel_output(
                self.diffmodel_trend(total_input[1], side_info, t, cfg_mask, timestep_emb, size_emb, context)
            )
            return predicted_seasonal + predicted_trend
        if self.save_attn:
            predicted, _ = self.diffmodel(total_input, side_info, t, cfg_mask, timestep_emb, size_emb, context)
            return predicted
        return self.diffmodel(total_input, side_info, t, cfg_mask, timestep_emb, size_emb, context)

    def _get_multi_res_confidence(self, batch_size, t=None, trend_prior=None):
        components = []

        if self.multi_res_dynamic_by_t and t is not None:
            if not torch.is_tensor(t):
                t = torch.tensor(t, device=self.device)
            t = t.float().reshape(-1)
            if t.numel() == 1:
                t = t.repeat(batch_size)
            step_conf = 1.0 - t / max(self.num_steps - 1, 1)
            components.append(step_conf.clamp(0.0, 1.0))

        if self.multi_res_dynamic_by_epoch:
            if self.total_epochs <= 1:
                epoch_conf = 1.0
            else:
                epoch_conf = float(self.current_epoch) / float(self.total_epochs - 1)
            components.append(torch.full((batch_size,), epoch_conf, device=self.device))

        if self.multi_res_dynamic_by_trend and trend_prior is not None:
            if trend_prior.dim() == 3:
                strength = trend_prior[:, :, 1].mean(dim=1).clamp(min=0.5, max=1.5)
                volatility = trend_prior[:, :, 2].mean(dim=1).clamp(min=0.0, max=1.0)
            else:
                strength = trend_prior[:, 1].clamp(min=0.5, max=1.5)
                volatility = trend_prior[:, 2].clamp(min=0.0, max=1.0)
            strength_conf = (strength - 0.5) / 1.0
            stability_conf = 1.0 - volatility
            trend_conf = 0.5 * (strength_conf + stability_conf)
            components.append(trend_conf.clamp(0.0, 1.0))

        if len(components) == 0:
            return torch.full((batch_size,), 0.5, device=self.device)

        return torch.stack(components, dim=0).mean(dim=0)

    def _get_multi_res_horizon_weights(self, horizons, batch_size, t=None, trend_prior=None):
        if len(horizons) <= 1:
            return torch.ones((batch_size, len(horizons)), device=self.device)

        base_weights = torch.ones((batch_size, len(horizons)), device=self.device)
        if self.multi_res_dynamic:
            horizon_tensor = torch.tensor(horizons, device=self.device, dtype=torch.float32)
            confidence = self._get_multi_res_confidence(batch_size, t=t, trend_prior=trend_prior).unsqueeze(1)
            horizon_pos = (horizon_tensor - horizon_tensor.min()) / max((horizon_tensor.max() - horizon_tensor.min()).item(), 1.0)
            horizon_pos = horizon_pos.unsqueeze(0).expand(batch_size, -1)

            min_w = min(max(self.multi_res_dynamic_min_weight, 0.0), 1.0)
            base_weights = min_w + (1.0 - min_w) * (1.0 - torch.abs(horizon_pos - confidence))

        if self.multi_res_difficulty_weight <= 0:
            return base_weights.clamp(min=1e-6)

        active_indices = [
            self.multi_res_horizon_to_index[horizon]
            for horizon in horizons
            if horizon in self.multi_res_horizon_to_index
        ]
        if len(active_indices) != len(horizons):
            return base_weights.clamp(min=1e-6)

        if self.multi_res_reliability_weight > 0 and self.multi_res_reliability.numel() >= len(self.multi_res_horizons):
            reliability = self.multi_res_reliability[active_indices].detach().clamp(min=0.0, max=1.0)
            reliability = reliability.unsqueeze(0).expand(batch_size, -1)
            rel_mix = float(min(max(self.multi_res_reliability_weight, 0.0), 1.0))
            base_weights = base_weights * ((1.0 - rel_mix) + rel_mix * reliability)

        difficulty = self.multi_res_difficulty_ema[active_indices].detach().clamp(min=1e-6)
        if self.multi_res_difficulty_inverse:
            difficulty = (difficulty.mean().clamp(min=1e-6) / difficulty).pow(max(float(self.multi_res_difficulty_gamma), 0.0))
        if self.multi_res_group_balance:
            group_to_indices = {}
            for idx, horizon in enumerate(horizons):
                group_name = self._get_horizon_group(horizon)
                group_to_indices.setdefault(group_name, []).append(idx)
            balanced = torch.ones_like(difficulty)
            active_groups = [group_indices for group_indices in group_to_indices.values() if len(group_indices) > 0]
            num_groups = max(len(active_groups), 1)
            total_horizons = max(len(horizons), 1)
            for group_indices in active_groups:
                group_tensor = difficulty[group_indices]
                group_tensor = group_tensor / group_tensor.mean().clamp(min=1e-6)
                scale = float(total_horizons) / float(num_groups * len(group_indices))
                balanced[group_indices] = group_tensor * scale
            difficulty = balanced
        else:
            difficulty = difficulty / difficulty.mean().clamp(min=1e-6)
        difficulty = difficulty.unsqueeze(0).expand(batch_size, -1)
        mix = float(min(max(self.multi_res_difficulty_weight, 0.0), 1.0))
        weights = (1.0 - mix) * base_weights + mix * difficulty
        if self.multi_res_group_max_ratio > 0 and len(horizons) > 1:
            group_to_indices = {}
            for idx, horizon in enumerate(horizons):
                group_name = self._get_horizon_group(horizon)
                group_to_indices.setdefault(group_name, []).append(idx)
            active_groups = [group_indices for group_indices in group_to_indices.values() if len(group_indices) > 0]
            if len(active_groups) > 1:
                group_means = [weights[:, group_indices].mean(dim=1) for group_indices in active_groups]
                group_means_tensor = torch.stack(group_means, dim=1)
                min_group_mean = group_means_tensor.min(dim=1, keepdim=True).values.clamp(min=1e-6)
                upper_bound = min_group_mean * max(self.multi_res_group_max_ratio, 1.0)
                for group_indices, group_mean in zip(active_groups, group_means):
                    scale = torch.minimum(torch.ones_like(group_mean), upper_bound.squeeze(1) / group_mean.clamp(min=1e-6))
                    weights[:, group_indices] = weights[:, group_indices] * scale.unsqueeze(1)
        return weights.clamp(min=1e-6)

    def _update_multi_res_difficulty(self, horizon, loss_value):
        if not self.training:
            return
        if self.multi_res_difficulty_weight <= 0:
            return
        horizon_index = self.multi_res_horizon_to_index.get(int(horizon))
        if horizon_index is None:
            return
        alpha = float(min(max(self.multi_res_ema_alpha, 0.0), 1.0))
        detached = loss_value.detach().float()
        with torch.no_grad():
            if alpha <= 0:
                self.multi_res_difficulty_ema[horizon_index] = detached
            else:
                self.multi_res_difficulty_ema[horizon_index].mul_(1.0 - alpha).add_(alpha * detached)

    def _calc_multi_res_loss(self, observed_data, predicted, target_mask, t=None, trend_prior=None):
        if self.pred_len <= 0:
            return torch.zeros((), device=observed_data.device)
        horizons = self._get_active_multi_res_horizons()
        if len(horizons) == 0:
            return torch.zeros((), device=observed_data.device)
        batch_size = observed_data.shape[0]
        horizon_weights = self._get_multi_res_horizon_weights(horizons, batch_size, t=t, trend_prior=trend_prior)
        weighted_loss_sum = torch.zeros((batch_size,), device=observed_data.device)
        weight_sum = torch.zeros((batch_size,), device=observed_data.device)
        for h_idx, h in enumerate(horizons):
            if h <= 0:
                continue
            horizon_mask = torch.zeros_like(target_mask)
            if self.multi_res_segment_loss:
                prev_h = int(horizons[h_idx - 1]) if h_idx > 0 else 0
                start = int(self.lookback_len + prev_h)
            else:
                start = int(self.lookback_len)
            end = int(self.lookback_len + h)
            if end <= start:
                continue
            horizon_mask[:, :, start:end] = 1.0
            horizon_mask = horizon_mask * target_mask
            num_eval = horizon_mask.sum(dim=(1, 2))
            valid = num_eval > 0
            if not valid.any():
                continue
            residual = (observed_data - predicted) * horizon_mask
            if self.multi_res_use_huber:
                full_index = self.multi_res_horizon_to_index.get(int(h))
                if full_index is None or full_index >= len(self.multi_res_huber_deltas):
                    delta = float(self.multi_res_huber_delta)
                else:
                    delta = float(self.multi_res_huber_deltas[full_index])
                abs_res = residual.abs()
                huber = torch.where(
                    abs_res <= delta,
                    0.5 * residual ** 2,
                    delta * abs_res - 0.5 * (delta ** 2),
                )
                loss_h = huber.sum(dim=(1, 2)) / num_eval.clamp(min=1.0)
            else:
                loss_h = (residual ** 2).sum(dim=(1, 2)) / num_eval.clamp(min=1.0)
            weight_h = horizon_weights[:, h_idx] * valid.float()
            weighted_loss_sum = weighted_loss_sum + weight_h * loss_h
            weight_sum = weight_sum + weight_h
            if valid.any():
                self._update_multi_res_difficulty(h, loss_h[valid].mean())
        valid_samples = weight_sum > 0
        if not valid_samples.any():
            return torch.zeros((), device=observed_data.device)
        return (weighted_loss_sum[valid_samples] / weight_sum[valid_samples]).mean()

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  
        else:
            cond_obs = cond_mask * observed_data
            noisy_target = noisy_data.unsqueeze(1) 
            if self.decomp:
                res, moving_mean = self.decomposition(cond_obs) 
                res, moving_mean = res.unsqueeze(1), moving_mean.unsqueeze(1) 
                res_input = torch.cat([res, noisy_target], dim=1)  
                moving_mean_input = torch.cat([moving_mean, noisy_target], dim=1) 
                total_input = [res_input, moving_mean_input]
            else:
                cond_obs = cond_obs.unsqueeze(1) 
                total_input = torch.cat([cond_obs, noisy_target], dim=1) 

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples, guide_w=None, timesteps=None, timestep_emb=None, size_emb=None, context=None, pattern_text_pooled=None, pattern_text_mask=None, pattern_text_evidence_vec=None):
        B, K, L = observed_data.shape
        if self.ddim:
            if self.sample_method == 'linear':
                a = self.num_steps // self.sample_steps
                time_steps = np.asarray(list(range(0, self.num_steps, a)))
            elif self.sample_method == "quad":
                time_steps = (np.linspace(0, np.sqrt(self.num_steps * 0.8), self.sample_steps) ** 2).astype(np.int)
            else:
                raise NotImplementedError(f"sampling method {self.sample_method} is not implemented!")
            time_steps = time_steps + 1
            time_steps_prev = np.concatenate([[0], time_steps[:-1]])
        else:
            self.sample_steps = self.num_steps
        if not self.noise_esti:
            means = torch.sum(observed_data*cond_mask, dim=2, keepdim=True) / torch.sum(cond_mask, dim=2, keepdim=True)
            stdev = torch.sqrt(torch.sum((observed_data - means) ** 2 * cond_mask, dim=2, keepdim=True) / (torch.sum(cond_mask, dim=2, keepdim=True) - 1) + 1e-5)
            observed_data = (observed_data - means) / stdev

        pattern_baseline = None
        diffusion_observed_data = observed_data
        if self.pattern_residual_diffusion and not self.noise_esti:
            pattern_guidance_scale = self.guide_w_default if guide_w is None else float(guide_w)
            pattern = self._compute_pattern_outputs(
                observed_data,
                cond_mask,
                text_pooled=pattern_text_pooled,
                text_mask=pattern_text_mask,
                text_evidence_vec=pattern_text_evidence_vec,
                guidance_scale=pattern_guidance_scale,
            )
            pattern_baseline = pattern["baseline"]
            diffusion_observed_data = observed_data - pattern_baseline
        
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        cfg_mask = None

        for i in range(n_samples):
            if self.is_unconditional == True:
                noisy_obs = diffusion_observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(diffusion_observed_data)
            for t in range(self.sample_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1) 
                else:
                    if self.decomp:
                        cond_obs = cond_mask * diffusion_observed_data
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1) # (B, 1, K, L)
                        res, moving_mean = self.decomposition(cond_obs) # (B, K, L), (B, K, L)
                        res, moving_mean = res.unsqueeze(1), moving_mean.unsqueeze(1) # (B, 1, K, L), (B, 1, K, L)
                        res_input = torch.cat([res, noisy_target], dim=1)  # (B,2,K,L)
                        moving_mean_input = torch.cat([moving_mean, noisy_target], dim=1)  # (B,2,K,L)
                        predicted_seasonal = self._unwrap_diffmodel_output(
                            self.diffmodel_sesonal(res_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, context)
                        ) # (2*B, K, L)
                        predicted_trend = self._unwrap_diffmodel_output(
                            self.diffmodel_trend(moving_mean_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, context)
                        ) # (2*B, K, L)
                        predicted = predicted_seasonal + predicted_trend # (2*B, K, L)
                    else:
                        cond_obs = (cond_mask * diffusion_observed_data).unsqueeze(1) # (B, 1, K, L)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1) # (B, 1, K, L)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)
                        if self.save_attn:
                            predicted, attn = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, context) # (2*B, K, L)
                        else:
                            predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, context) # (2*B, K, L)

                if self.noise_esti:
                    # noise prediction
                    if not self.ddim:
                        coeff1 = 1 / self.alpha_hat[t] ** 0.5
                        coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                        current_sample = coeff1 * (current_sample - coeff2 * predicted) # (B, K, L)
                        if t > 0:
                            noise = torch.randn_like(current_sample)
                            sigma = (
                                (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                            current_sample += sigma * noise
                    else:
                        tau, tau_prev = time_steps[t], time_steps_prev[t]
                        current_sample = (
                            torch.sqrt(self.alpha[tau_prev] / self.alpha[tau]) * current_sample +
                            (torch.sqrt(1 - self.alpha[tau_prev]) - torch.sqrt(
                                (self.alpha[tau_prev] * (1 - self.alpha[tau])) / self.alpha[tau])) * predicted
                        )
                else:
                    if not self.ddim:
                        if t > 1:
                            # data prediction
                            coeff1 = (self.alpha_hat[t] ** 0.5 * (1 - self.alpha[t-1])) / (1 - self.alpha[t])
                            coeff2 = (self.alpha[t-1] ** 0.5 * self.beta[t]) / (1 - self.alpha[t])
                            current_sample = coeff1 * current_sample + coeff2 * predicted # (B, K, L)
                            
                            if t > 2:
                                noise = torch.randn_like(current_sample)
                                sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                                ) ** 0.5
                                current_sample += sigma * noise
                    else:
                        tau, tau_prev = time_steps[t], time_steps_prev[t]
                        aaa_ = (1-self.alpha[tau_prev])/(1-self.alpha[tau]) ** 0.5
                        current_sample = (
                            aaa_ * current_sample +
                            ((self.alpha[tau_prev])**0.5 - (self.alpha[tau])**0.5 * aaa_) * predicted
                        )

            sample = current_sample.detach()
            if pattern_baseline is not None:
                sample = sample + pattern_baseline.detach()
            imputed_samples[:, i] = sample
            if self.timestep_branch and timesteps is not None:
                predicted_from_timestep = self.timestep_pred(timesteps)
                imputed_samples[:, i] = 0.9 * imputed_samples[:, i] + 0.1 * predicted_from_timestep.detach()
            if not self.noise_esti:
                imputed_samples[:, i] = imputed_samples[:, i] * stdev + means
        if self.save_attn:
            return imputed_samples, attn 
        else:
            return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)): 
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim, window_lens):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device, window_lens)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]
        

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        batch_size = observed_data.shape[0]
        text_mask = batch["text_mark"].to(self.device).float().reshape(-1).clamp(0.0, 1.0)
        text_quality_raw = batch.get("text_quality_raw")
        if text_quality_raw is not None:
            text_quality_raw = text_quality_raw.to(self.device).float().reshape(-1)
        else:
            text_quality_raw = text_mask
        trend_prior_num = batch.get("trend_prior_num", batch.get("trend_prior"))
        if trend_prior_num is None:
            trend_prior_num = torch.zeros((observed_data.shape[0], 3), device=self.device)
        else:
            trend_prior_num = trend_prior_num.to(self.device).float()
        trend_prior_text = batch.get("trend_prior_text")
        if trend_prior_text is None:
            trend_prior_text = trend_prior_num.clone()
        else:
            trend_prior_text = trend_prior_text.to(self.device).float()
        text_evidence_vec = batch.get("text_evidence_vec")
        if text_evidence_vec is not None:
            text_evidence_vec = text_evidence_vec.to(self.device).float().reshape(observed_data.shape[0], -1)
        else:
            text_evidence_vec = torch.stack(
                [
                    text_quality_raw,
                    text_quality_raw,
                    text_quality_raw,
                    text_quality_raw,
                    text_quality_raw,
                    text_quality_raw,
                    text_quality_raw,
                ],
                dim=1,
            )
        if self.timestep_emb_cat or self.timestep_branch:
            timesteps = batch["timesteps"].to(self.device).float()
            timesteps = timesteps.permute(0, 2, 1)
        else:
            timesteps = None
        if self.with_texts:
            texts = batch.get("texts", batch.get("text_raw"))
            text_event_texts = self._reshape_text_event_batch(batch.get("text_event_texts"), batch_size)
            text_event_source_ids = batch.get("text_event_source_ids")
            text_event_time_deltas = batch.get("text_event_time_deltas")
            text_event_quality_feats = batch.get("text_event_quality_feats")
            text_event_mask = batch.get("text_event_mask")
            if text_event_source_ids is not None:
                text_event_source_ids = text_event_source_ids.to(self.device).long()
            if text_event_time_deltas is not None:
                text_event_time_deltas = text_event_time_deltas.to(self.device).float()
            if text_event_quality_feats is not None:
                text_event_quality_feats = text_event_quality_feats.to(self.device).float()
            if text_event_mask is not None:
                text_event_mask = text_event_mask.to(self.device).float()
        else:
            texts = None
            text_event_texts = None
            text_event_source_ids = None
            text_event_time_deltas = None
            text_event_quality_feats = None
            text_event_mask = None

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id,
            timesteps,
            texts,
            text_mask,
            trend_prior_num,
            trend_prior_text,
            text_evidence_vec,
            text_event_texts,
            text_event_source_ids,
            text_event_time_deltas,
            text_event_quality_feats,
            text_event_mask,
        )        

    def _unpack_forecasting_batch(self, data):
        defaults = {
            "feature_id": None,
            "timesteps": None,
            "texts": None,
            "text_mask": None,
            "trend_prior_num": None,
            "trend_prior_text": None,
            "text_evidence_vec": None,
            "text_event_texts": None,
            "text_event_source_ids": None,
            "text_event_time_deltas": None,
            "text_event_quality_feats": None,
            "text_event_mask": None,
        }
        if len(data) >= 18:
            (
                observed_data,
                observed_mask,
                observed_tp,
                gt_mask,
                _,
                _,
                defaults["feature_id"],
                defaults["timesteps"],
                defaults["texts"],
                defaults["text_mask"],
                defaults["trend_prior_num"],
                defaults["trend_prior_text"],
                defaults["text_evidence_vec"],
                defaults["text_event_texts"],
                defaults["text_event_source_ids"],
                defaults["text_event_time_deltas"],
                defaults["text_event_quality_feats"],
                defaults["text_event_mask"],
            ) = data
        elif len(data) == 13:
            (
                observed_data,
                observed_mask,
                observed_tp,
                gt_mask,
                _,
                _,
                defaults["feature_id"],
                defaults["timesteps"],
                defaults["texts"],
                defaults["text_mask"],
                defaults["trend_prior_num"],
                defaults["trend_prior_text"],
                defaults["text_evidence_vec"],
            ) = data
        elif len(data) == 12:
            (
                observed_data,
                observed_mask,
                observed_tp,
                gt_mask,
                _,
                _,
                defaults["feature_id"],
                defaults["timesteps"],
                defaults["texts"],
                defaults["text_mask"],
                defaults["trend_prior_num"],
                defaults["trend_prior_text"],
            ) = data
        elif len(data) == 11:
            (
                observed_data,
                observed_mask,
                observed_tp,
                gt_mask,
                _,
                _,
                defaults["feature_id"],
                defaults["timesteps"],
                defaults["texts"],
                defaults["text_mask"],
                defaults["trend_prior_num"],
            ) = data
            defaults["trend_prior_text"] = defaults["trend_prior_num"]
        elif len(data) == 10:
            (
                observed_data,
                observed_mask,
                observed_tp,
                gt_mask,
                _,
                _,
                defaults["feature_id"],
                defaults["timesteps"],
                defaults["texts"],
                defaults["text_mask"],
            ) = data
        else:
            (
                observed_data,
                observed_mask,
                observed_tp,
                gt_mask,
                _,
                _,
            ) = data
        return observed_data, observed_mask, observed_tp, gt_mask, defaults

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask
    
    def get_timestep_info(self, timesteps):
        timestep_emb = self.timestep_emb(timesteps.transpose(1, 2)).transpose(1, 2)
        timestep_emb = timestep_emb.unsqueeze(2).expand(-1, -1, self.target_dim, -1) 
        return timestep_emb
    
    def get_relative_size_info(self, observed_data):
        B, K, L = observed_data.shape

        size_emb = observed_data[:, :, :self.lookback_len].clone().unsqueeze(3).expand(-1, -1, -1, self.lookback_len) - \
            observed_data[:, :, :self.lookback_len].clone().unsqueeze(2).expand(-1, -1, self.lookback_len, -1) 
        size_emb = self.relative_size_emb(size_emb)
        size_emb = size_emb.permute(0, 3, 1, 2)
        size_emb = torch.cat([size_emb, torch.zeros((B, self.diff_channels, K, self.pred_len)).to(observed_data.device)], dim=-1) 
        return size_emb

    def get_trend_step_ratio(self, step_index, time_steps=None):
        if self.ddim and time_steps is not None:
            current_step = float(time_steps[step_index])
        else:
            current_step = float(step_index)
        denom = max(self.num_steps - 1, 1)
        ratio = 1.0 - current_step / denom
        ratio = ratio ** self.trend_cfg_power
        floor = max(self.trend_time_floor, 0.0)
        if floor > 0.0:
            ratio = floor + (1.0 - floor) * ratio
        return ratio

    def _to_samplewise_weight(self, value, batch_size, device):
        if torch.is_tensor(value):
            value = value.to(device).float().reshape(-1)
            if value.numel() == 1:
                value = value.repeat(batch_size)
            return value
        return torch.full((batch_size,), float(value), device=device)

    def _reshape_text_event_batch(self, event_texts, batch_size):
        if event_texts is None:
            return None
        if isinstance(event_texts, (list, tuple)):
            if len(event_texts) == 0:
                return [[] for _ in range(batch_size)]
            if isinstance(event_texts[0], str):
                return [[str(item)] for item in event_texts]
            transposed = list(zip(*event_texts))
            return [[str(item) for item in row] for row in transposed]
        return None

    def _encode_text_source(self, text):
        return self._encode_text_source_with_options(text, max_length=self.text_max_length, return_sequence=False)

    def _encode_text_source_with_options(self, text, max_length=None, return_sequence=True):
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, (list, tuple)):
            texts = [str(item) if item is not None else "NA" for item in text]
        else:
            texts = [str(text)]

        max_length = int(max_length or self.text_max_length)
        chunk_size = max(int(self.text_encode_batch_size), 1)
        encoded_chunks = []
        pooled_chunks = []
        input_id_chunks = []
        attention_chunks = []

        for start in range(0, len(texts), chunk_size):
            text_chunk = texts[start:start + chunk_size]
            token_input = self.tokenizer(
                text_chunk,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            ).to(self.device)
            with torch.inference_mode():
                encoded = self.text_encoder(**token_input).last_hidden_state
            attention_mask = token_input["attention_mask"].unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1.0)
            if return_sequence:
                encoded_chunks.append(encoded.detach())
            pooled_chunks.append(pooled.detach())
            input_id_chunks.append(token_input["input_ids"].detach())
            attention_chunks.append(token_input["attention_mask"].detach())

        encoded_text = torch.cat(encoded_chunks, dim=0) if return_sequence else None
        pooled_text = torch.cat(pooled_chunks, dim=0)
        token_input = {
            "input_ids": torch.cat(input_id_chunks, dim=0),
            "attention_mask": torch.cat(attention_chunks, dim=0),
        }
        return encoded_text, pooled_text, token_input

    def get_side_info(self, observed_tp, cond_mask, feature_id=None, timesteps=None, texts=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim) 
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1) 

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            ) 
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else: 
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1) 

        side_info = torch.cat([time_embed, feature_embed], dim=-1) 
        side_info = side_info.permute(0, 3, 2, 1) 

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1) 
            side_info = torch.cat([side_info, side_mask], dim=1) 
    

        return side_info

    def forward(self, batch, is_train=1):
        data = self.process_data(batch)
        observed_data, observed_mask, observed_tp, gt_mask, unpacked = self._unpack_forecasting_batch(data)
        feature_id = unpacked["feature_id"]
        timesteps = unpacked["timesteps"]
        texts = unpacked["texts"]
        text_mask = unpacked["text_mask"]
        trend_prior_num = unpacked["trend_prior_num"]
        trend_prior_text = unpacked["trend_prior_text"]
        text_evidence_vec = unpacked["text_evidence_vec"]
        text_event_texts = unpacked["text_event_texts"]
        text_event_source_ids = unpacked["text_event_source_ids"]
        text_event_time_deltas = unpacked["text_event_time_deltas"]
        text_event_quality_feats = unpacked["text_event_quality_feats"]
        text_event_mask = unpacked["text_event_mask"]
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id, timesteps, texts)

        if self.timestep_emb_cat:
            timestep_emb = self.get_timestep_info(timesteps)
        else:
            timestep_emb = None

        if self.relative_size_emb_cat:
            size_emb = self.get_relative_size_info(observed_data)
        else:
            size_emb = None

        if self.with_texts:
            _, text_pooled, _ = self._encode_text_source(texts)
        else:
            text_pooled = None

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(
            observed_data,
            cond_mask,
            observed_mask,
            side_info,
            is_train,
            timesteps=timesteps,
            timestep_emb=timestep_emb,
            size_emb=size_emb,
            context=None,
            trend_prior=trend_prior_num,
            pattern_text_pooled=text_pooled,
            pattern_text_mask=text_mask,
            pattern_text_evidence_vec=text_evidence_vec,
        )

    def evaluate(self, batch, n_samples, guide_w):
        data = self.process_data(batch)
        observed_data, observed_mask, observed_tp, gt_mask, unpacked = self._unpack_forecasting_batch(data)
        feature_id = unpacked["feature_id"]
        timesteps = unpacked["timesteps"]
        texts = unpacked["texts"]
        text_mask = unpacked["text_mask"]
        trend_prior_num = unpacked["trend_prior_num"]
        trend_prior_text = unpacked["trend_prior_text"]
        text_evidence_vec = unpacked["text_evidence_vec"]
        text_event_texts = unpacked["text_event_texts"]
        text_event_source_ids = unpacked["text_event_source_ids"]
        text_event_time_deltas = unpacked["text_event_time_deltas"]
        text_event_quality_feats = unpacked["text_event_quality_feats"]
        text_event_mask = unpacked["text_event_mask"]

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask, timesteps=timesteps, texts=texts)

            if self.timestep_emb_cat:
                timestep_emb = self.get_timestep_info(timesteps)
            else:
                timestep_emb = None

            if self.relative_size_emb_cat:
                size_emb = self.get_relative_size_info(observed_data)
            else:
                size_emb = None

            tokens = None
            if self.with_texts:
                _, text_pooled, token_input = self._encode_text_source(texts)
                if self.save_token:
                    tokens = self.tokenizer.batch_decode(token_input['input_ids'])
            else:
                text_pooled = None
            if self.save_attn:
                samples, attn = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=None, pattern_text_pooled=text_pooled, pattern_text_mask=text_mask, pattern_text_evidence_vec=text_evidence_vec)
            else:
                samples = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=None, pattern_text_pooled=text_pooled, pattern_text_mask=text_mask, pattern_text_evidence_vec=text_evidence_vec)

        if self.save_attn:
            if self.save_token:
                return samples, observed_data, target_mask, observed_mask, observed_tp, attn, tokens
            else:
                return samples, observed_data, target_mask, observed_mask, observed_tp, attn
        else:
            return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
