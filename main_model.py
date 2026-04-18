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
        dataset_cfg = {}
        dataset_cfg.update(config.get("data", {}))
        dataset_cfg.update(config.get("dataset", {}))
        self.llm = config["model"]["llm"]
        self.domain = config["model"]["domain"]
        self.save_attn = config["model"]["save_attn"]
        self.save_token = config["model"]["save_token"]
        self.text_quality_gate = bool(config["model"].get("text_quality_gate", True))
        self.text_quality_min_scale = float(config["model"].get("text_quality_min_scale", 0.0))
        quality_weights = config["model"].get("text_quality_weights", [0.5, 0.3, 0.2])
        if not isinstance(quality_weights, (list, tuple)) or len(quality_weights) != 3:
            quality_weights = [0.5, 0.3, 0.2]
        quality_weights = np.asarray(quality_weights, dtype=np.float32)
        self.text_quality_weights = quality_weights / max(float(np.sum(quality_weights)), 1e-6)
        self.text_quality_drop_threshold = float(config["model"].get("text_quality_drop_threshold", 0.3))
        self.text_quality_mid_threshold = float(config["model"].get("text_quality_mid_threshold", 0.6))
        self.text_use_ret_in_context = bool(config["model"].get("text_use_ret_in_context", False))
        self.text_use_cot_in_context = bool(config["model"].get("text_use_cot_in_context", False))
        self.text_trend_only_guidance = bool(config["model"].get("text_trend_only_guidance", True))
        self.text_trend_ret_scale = float(config["model"].get("text_trend_ret_scale", 0.5))
        self.text_trend_cot_scale = float(config["model"].get("text_trend_cot_scale", 0.3))
        self.text_trend_raw_weight = float(config["model"].get("text_trend_raw_weight", 1.0))
        self.text_trend_ret_weight = float(config["model"].get("text_trend_ret_weight", 0.35))
        self.text_trend_cot_weight = float(config["model"].get("text_trend_cot_weight", 0.15))
        self.text_numeric_align_gamma = float(config["model"].get("text_numeric_align_gamma", 2.0))
        self.multi_res_trend_source = str(config["model"].get("multi_res_trend_source", "numeric_only")).lower()
        self.text_aug_max_ratio = float(config["model"].get("text_aug_max_ratio", 0.3))
        self.coverage_power = float(config["model"].get("coverage_power", 2.0))
        self.coverage_cfg_boost = float(config["model"].get("coverage_cfg_boost", 0.15))
        self.reliability_min = float(config["model"].get("reliability_min", 0.1))
        self.guide_reliability_power = float(config["model"].get("guide_reliability_power", 1.0))
        self.semantic_dim = int(config["model"].get("semantic_dim", 6))
        self.event_quality_dim = int(config["model"].get("event_quality_dim", 6))
        self.event_quality_beta = float(config["model"].get("event_quality_beta", 4.0))
        self.semantic_trend_scale = float(config["model"].get("semantic_trend_scale", 0.25))
        self.text_recency_tau_days = float(dataset_cfg.get("text_recency_tau_days", 14.0))
        self.text_coverage_kappa = float(dataset_cfg.get("text_coverage_kappa", 3.0))
        self.text_max_length = int(config["model"].get("text_max_length", 192))
        self.event_text_max_length = int(config["model"].get("event_text_max_length", min(self.text_max_length, 96)))
        self.text_encode_batch_size = int(config["model"].get("text_encode_batch_size", 8))
        self.use_gate_min = float(config["model"].get("use_gate_min", 0.0))
        self.strength_gate_min = float(config["model"].get("strength_gate_min", 0.0))
        self.strength_use_mix_floor = float(config["model"].get("strength_use_mix_floor", 0.5))
        self.horizon_strength_bias = float(config["model"].get("horizon_strength_bias", 0.0))
        self.text_context_ratio_min = float(config["model"].get("text_context_ratio_min", config["model"].get("text_context_strength_scale", 0.1)))
        self.text_context_ratio_max = float(config["model"].get("text_context_ratio_max", 0.6))
        self.text_context_max_base = float(config["model"].get("text_context_max_base", min(config["model"].get("text_context_strength_max", 0.03), 1.0)))
        self.text_context_max_boost = float(config["model"].get("text_context_max_boost", 0.07))
        self.text_context_horizon_bias = float(config["model"].get("text_context_horizon_bias", config["model"].get("horizon_strength_bias", 0.0)))
        self.text_guide_ratio_max = float(config["model"].get("text_guide_ratio_max", 0.35))
        self.text_guide_max_base = float(config["model"].get("text_guide_max_base", 0.02))
        self.text_guide_max_boost = float(config["model"].get("text_guide_max_boost", 0.05))
        self.text_guide_quality_power = float(config["model"].get("text_guide_quality_power", 0.5))
        self.text_guide_step_low = float(config["model"].get("text_guide_step_low", 0.2))
        self.text_guide_step_high = float(config["model"].get("text_guide_step_high", 0.8))
        self.text_guide_step_k = float(config["model"].get("text_guide_step_k", 12.0))
        self.use_gate_warmup_epochs = int(config["model"].get("use_gate_warmup_epochs", max(1, int(train_cfg.get("lr_warmup_epochs", 0)))))
        self.strength_gate_warmup_epochs = int(config["model"].get("strength_gate_warmup_epochs", max(self.use_gate_warmup_epochs + 1, int(0.2 * max(int(train_cfg.get("epochs", 1)), 1)))))
        self.text_use_weight = float(train_cfg.get("text_use_weight", train_cfg.get("text_benefit_weight", 0.0)))
        self.text_use_margin = float(train_cfg.get("text_use_margin", 0.02))
        self.text_strength_weight = float(train_cfg.get("text_strength_weight", train_cfg.get("text_uplift_weight", train_cfg.get("text_notext_fallback_weight", 0.0))))
        self.text_strength_tau = float(train_cfg.get("text_strength_tau", train_cfg.get("text_uplift_tau", 0.1)))
        self.detach_text_baselines = bool(train_cfg.get("detach_text_baselines", True))
        self.text_uplift_weight = self.text_strength_weight
        self.text_uplift_tau = self.text_strength_tau
        self.latest_reliability_stats = {}

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
        self.text_consistency_weight = float(train_cfg.get("text_consistency_weight", 0.0))
        self.text_benefit_weight = self.text_use_weight
        self.text_aug_benefit_weight = float(train_cfg.get("text_aug_benefit_weight", 0.0))
        self.text_aug_reg_weight = float(train_cfg.get("text_aug_reg_weight", 0.0))
        self.auxiliary_loss_max_ratio = float(train_cfg.get("auxiliary_loss_max_ratio", 0.0))
        self.current_epoch = 0
        self.total_epochs = max(int(train_cfg.get("epochs", 1)), 1)
        self.multi_res_horizons = self._sanitize_multi_res_horizons(self.multi_res_horizons)
        self.multi_res_horizon_to_index = {
            int(horizon): idx for idx, horizon in enumerate(self.multi_res_horizons)
        }
        difficulty_size = max(len(self.multi_res_horizons), 1)
        self.register_buffer(
            "multi_res_difficulty_ema",
            torch.ones(difficulty_size, dtype=torch.float32),
        )
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
            benefit_hidden_dim = int(config["model"].get("text_benefit_hidden_dim", 128))
            event_source_embed_dim = int(config["model"].get("event_source_embed_dim", 4))
            self.text_benefit_head = nn.Sequential(
                nn.Linear(text_hidden_dim + 7, benefit_hidden_dim),
                nn.LayerNorm(benefit_hidden_dim),
                nn.ReLU(),
                nn.Linear(benefit_hidden_dim, 1),
            )
            gate_hidden_dim = int(config["model"].get("reliability_hidden_dim", 32))
            evidence_dim = 7
            self.event_source_embed = nn.Embedding(3, event_source_embed_dim)
            self.event_semantic_proj = nn.Sequential(
                nn.Linear(text_hidden_dim + event_source_embed_dim + self.event_quality_dim, gate_hidden_dim),
                nn.LayerNorm(gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, self.semantic_dim),
            )
            self.semantic_proj = nn.Sequential(
                nn.Linear(text_hidden_dim, gate_hidden_dim),
                nn.LayerNorm(gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, self.semantic_dim),
            )
            self.semantic_to_trend = nn.Sequential(
                nn.Linear(self.semantic_dim, gate_hidden_dim),
                nn.LayerNorm(gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, self.pred_len * 3),
            )
            self.use_gate_head = nn.Sequential(
                nn.Linear(evidence_dim, gate_hidden_dim),
                nn.LayerNorm(gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, self.pred_len),
            )
            self.strength_gate_head = nn.Sequential(
                nn.Linear(evidence_dim + self.semantic_dim + 6, gate_hidden_dim),
                nn.LayerNorm(gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, self.pred_len),
            )
            nn.init.zeros_(self.use_gate_head[-1].weight)
            nn.init.constant_(self.use_gate_head[-1].bias, -1.5)
            nn.init.zeros_(self.strength_gate_head[-1].weight)
            nn.init.constant_(self.strength_gate_head[-1].bias, -2.0)
            aug_hidden_dim = int(config["model"].get("text_aug_hidden_dim", 128))
            aug_input_dim = text_hidden_dim * 2 + 9
            self.text_aug_gate_head = nn.Sequential(
                nn.Linear(aug_input_dim, aug_hidden_dim),
                nn.LayerNorm(aug_hidden_dim),
                nn.ReLU(),
                nn.Linear(aug_hidden_dim, 1),
            )
            self.text_aug_adapter = nn.Sequential(
                nn.Linear(aug_input_dim, aug_hidden_dim),
                nn.LayerNorm(aug_hidden_dim),
                nn.ReLU(),
                nn.Linear(aug_hidden_dim, text_hidden_dim),
            )

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
            "difficulty_ema": difficulty,
            "horizon_groups": [self._get_horizon_group(horizon) for horizon in active_horizons],
            "reliability": dict(self.latest_reliability_stats),
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

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, timesteps=None, timestep_emb=None, size_emb=None, context=None, trend_prior=None, text_mask=None, use_gate=None, context_raw=None, aug_gate=None, strength_gate=None
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=context, trend_prior=trend_prior, text_mask=text_mask, use_gate=use_gate, context_raw=context_raw, aug_gate=aug_gate, strength_gate=strength_gate
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, timesteps=None, timestep_emb=None, size_emb=None, context=None, trend_prior=None, text_mask=None, use_gate=None, context_raw=None, aug_gate=None, strength_gate=None, set_t=-1
    ):

        B, K, L = observed_data.shape
        if not self.noise_esti:
            means = torch.sum(observed_data*cond_mask, dim=2, keepdim=True) / torch.sum(cond_mask, dim=2, keepdim=True)
            stdev = torch.sqrt(torch.sum((observed_data - means) ** 2 * cond_mask, dim=2, keepdim=True) / (torch.sum(cond_mask, dim=2, keepdim=True) - 1) + 1e-5)
            observed_data = (observed_data - means) / stdev

        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

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
            residual = (observed_data - predicted) * target_mask 
        num_eval = target_mask.sum()
        main_loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        auxiliary_loss = torch.zeros((), device=observed_data.device)
        if (not self.noise_esti) and self.multi_res_loss_weight > 0 and len(self.multi_res_horizons) > 0:
            aux_loss = self._calc_multi_res_loss(observed_data, predicted, target_mask, t=t, trend_prior=trend_prior)
            auxiliary_loss = auxiliary_loss + self.multi_res_loss_weight * aux_loss
        needs_text_baseline = (
            (not self.noise_esti)
            and self.training
            and self.with_texts
            and (context is not None or context_raw is not None)
            and (
                (self.text_consistency_weight > 0 and text_mask is not None)
                or (self.text_use_weight > 0 and use_gate is not None)
                or (self.text_aug_benefit_weight > 0 and aug_gate is not None)
                or (self.text_aug_reg_weight > 0 and aug_gate is not None)
                or (self.text_strength_weight > 0 and strength_gate is not None)
            )
        )
        if needs_text_baseline:
            if self.detach_text_baselines:
                with torch.no_grad():
                    predicted_no_text = self._run_diffusion_model(
                        total_input,
                        side_info,
                        t,
                        cfg_mask,
                        timestep_emb=timestep_emb,
                        size_emb=size_emb,
                        context=None,
                    )
                    if self.timestep_branch and timesteps is not None:
                        predicted_from_timestep = self.timestep_pred(timesteps)
                        predicted_no_text = 0.9 * predicted_no_text + 0.1 * predicted_from_timestep
            else:
                predicted_no_text = self._run_diffusion_model(
                    total_input,
                    side_info,
                    t,
                    cfg_mask,
                    timestep_emb=timestep_emb,
                    size_emb=size_emb,
                    context=None,
                )
                if self.timestep_branch and timesteps is not None:
                    predicted_from_timestep = self.timestep_pred(timesteps)
                    predicted_no_text = 0.9 * predicted_no_text + 0.1 * predicted_from_timestep
            predicted_raw = None
            if context_raw is not None and (
                self.text_consistency_weight > 0
                or self.text_benefit_weight > 0
                or self.text_aug_benefit_weight > 0
                or self.text_aug_reg_weight > 0
            ):
                if self.detach_text_baselines:
                    with torch.no_grad():
                        predicted_raw = self._run_diffusion_model(
                            total_input,
                            side_info,
                            t,
                            cfg_mask,
                            timestep_emb=timestep_emb,
                            size_emb=size_emb,
                            context=context_raw,
                        )
                        if self.timestep_branch and timesteps is not None:
                            predicted_from_timestep = self.timestep_pred(timesteps)
                            predicted_raw = 0.9 * predicted_raw + 0.1 * predicted_from_timestep
                else:
                    predicted_raw = self._run_diffusion_model(
                        total_input,
                        side_info,
                        t,
                        cfg_mask,
                        timestep_emb=timestep_emb,
                        size_emb=size_emb,
                        context=context_raw,
                    )
                    if self.timestep_branch and timesteps is not None:
                        predicted_from_timestep = self.timestep_pred(timesteps)
                        predicted_raw = 0.9 * predicted_raw + 0.1 * predicted_from_timestep
            if (not self.detach_text_baselines) and self.text_consistency_weight > 0 and text_mask is not None and predicted_raw is not None:
                consistency_loss = self._calc_text_consistency_loss(
                    predicted_raw,
                    predicted_no_text,
                    target_mask,
                    text_mask,
                )
                auxiliary_loss = auxiliary_loss + self.text_consistency_weight * consistency_loss
            use_loss_weight = self.text_use_weight * self._get_gate_loss_scale("use")
            if use_loss_weight > 0 and use_gate is not None:
                use_loss = self._calc_text_use_loss(
                    predicted,
                    predicted_no_text,
                    observed_data,
                    target_mask,
                    use_gate,
                )
                auxiliary_loss = auxiliary_loss + use_loss_weight * use_loss
            if self.text_aug_benefit_weight > 0 and aug_gate is not None and predicted_raw is not None:
                aug_benefit_loss = self._calc_text_benefit_loss(
                    predicted,
                    predicted_raw,
                    observed_data,
                    target_mask,
                    aug_gate,
                )
                auxiliary_loss = auxiliary_loss + self.text_aug_benefit_weight * aug_benefit_loss
            if self.text_aug_reg_weight > 0 and aug_gate is not None and predicted_raw is not None:
                aug_reg_loss = self._calc_text_aug_reg_loss(
                    predicted,
                    predicted_raw,
                    target_mask,
                    aug_gate,
                )
                auxiliary_loss = auxiliary_loss + self.text_aug_reg_weight * aug_reg_loss
            strength_loss_weight = self.text_strength_weight * self._get_gate_loss_scale("strength")
            if strength_loss_weight > 0 and strength_gate is not None:
                strength_loss = self._calc_text_strength_loss(
                    predicted,
                    predicted_no_text,
                    observed_data,
                    target_mask,
                    strength_gate,
                )
                auxiliary_loss = auxiliary_loss + strength_loss_weight * strength_loss
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
            seasonal_context = None if self.text_trend_only_guidance else context
            trend_context = context
            predicted_seasonal = self._unwrap_diffmodel_output(
                self.diffmodel_sesonal(total_input[0], side_info, t, cfg_mask, timestep_emb, size_emb, seasonal_context)
            )
            predicted_trend = self._unwrap_diffmodel_output(
                self.diffmodel_trend(total_input[1], side_info, t, cfg_mask, timestep_emb, size_emb, trend_context)
            )
            return predicted_seasonal + predicted_trend
        if self.save_attn:
            predicted, _ = self.diffmodel(total_input, side_info, t, cfg_mask, timestep_emb, size_emb, context)
            return predicted
        return self.diffmodel(total_input, side_info, t, cfg_mask, timestep_emb, size_emb, context)

    def _calc_text_consistency_loss(self, predicted, predicted_no_text, target_mask, text_mask):
        diff = (predicted - predicted_no_text) * target_mask
        num_eval = target_mask.sum(dim=(1, 2)).clamp(min=1.0)
        per_sample = (diff ** 2).sum(dim=(1, 2)) / num_eval
        quality = text_mask.float().clamp(min=0.0, max=1.0)
        if quality.dim() > 1:
            quality = quality.mean(dim=1)
        else:
            quality = quality.reshape(-1)
        weights = 1.0 - quality
        valid = weights > 0
        if not valid.any():
            return torch.zeros((), device=predicted.device)
        return (per_sample[valid] * weights[valid]).sum() / weights[valid].sum().clamp(min=1e-6)

    def _calc_per_sample_forecast_loss(self, predicted, observed_data, target_mask):
        residual = (observed_data - predicted) * target_mask
        num_eval = target_mask.sum(dim=(1, 2)).clamp(min=1.0)
        return (residual ** 2).sum(dim=(1, 2)) / num_eval

    def _calc_per_horizon_forecast_loss(self, predicted, observed_data, target_mask):
        future_slice = slice(self.lookback_len, self.lookback_len + self.pred_len)
        future_residual = (observed_data - predicted)[:, :, future_slice] * target_mask[:, :, future_slice]
        future_mask = target_mask[:, :, future_slice]
        num_eval = future_mask.sum(dim=1).clamp(min=1.0)
        return (future_residual ** 2).sum(dim=1) / num_eval

    def _calc_text_benefit_loss(self, predicted, predicted_no_text, observed_data, target_mask, benefit_gate):
        text_loss = self._calc_per_sample_forecast_loss(predicted, observed_data, target_mask)
        base_loss = self._calc_per_sample_forecast_loss(predicted_no_text, observed_data, target_mask)
        target = (base_loss.detach() > text_loss.detach()).float()
        if benefit_gate.dim() > 1:
            gate = benefit_gate.float().mean(dim=1)
        else:
            gate = benefit_gate.reshape(-1).float()
        gate = gate.clamp(min=1e-4, max=1.0 - 1e-4)
        return F.binary_cross_entropy(gate, target)

    def _calc_text_use_loss(self, predicted, predicted_no_text, observed_data, target_mask, use_gate):
        gate = self._ensure_horizon_gate(use_gate)
        if gate is None:
            return torch.zeros((), device=predicted.device)
        text_loss = self._calc_per_horizon_forecast_loss(predicted, observed_data, target_mask)
        base_loss = self._calc_per_horizon_forecast_loss(predicted_no_text, observed_data, target_mask)
        margin = max(float(self.text_use_margin), 0.0)
        target = (text_loss.detach() <= (base_loss.detach() + margin)).float()
        gate = gate.float().clamp(min=1e-4, max=1.0 - 1e-4)
        return F.binary_cross_entropy(gate, target)

    def _calc_text_aug_reg_loss(self, predicted_aug, predicted_raw, target_mask, aug_gate):
        diff = (predicted_aug - predicted_raw) * target_mask
        num_eval = target_mask.sum(dim=(1, 2)).clamp(min=1.0)
        per_sample = (diff ** 2).sum(dim=(1, 2)) / num_eval
        if aug_gate.dim() > 1:
            aug_weight = aug_gate.float().mean(dim=1)
        else:
            aug_weight = aug_gate.reshape(-1).float()
        weights = 1.0 - aug_weight.clamp(min=0.0, max=1.0)
        valid = weights > 0
        if not valid.any():
            return torch.zeros((), device=predicted_aug.device)
        return (per_sample[valid] * weights[valid]).sum() / weights[valid].sum().clamp(min=1e-6)

    def _calc_text_strength_loss(self, predicted, predicted_no_text, observed_data, target_mask, strength_gate):
        gate = self._ensure_horizon_gate(strength_gate)
        if gate is None:
            return torch.zeros((), device=predicted.device)
        text_loss = self._calc_per_horizon_forecast_loss(predicted, observed_data, target_mask)
        base_loss = self._calc_per_horizon_forecast_loss(predicted_no_text, observed_data, target_mask)
        tau = max(self.text_strength_tau, 1e-6)
        usable_target = (text_loss.detach() <= (base_loss.detach() + max(float(self.text_use_margin), 0.0))).float()
        target = torch.sigmoid((base_loss.detach() - text_loss.detach()) / tau) * usable_target
        gate = gate.float()
        valid = gate > 0
        if not valid.any():
            return torch.zeros((), device=predicted.device)
        gate = gate[valid].clamp(min=1e-4, max=1.0 - 1e-4)
        return F.binary_cross_entropy(gate, target[valid])

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

        difficulty = self.multi_res_difficulty_ema[active_indices].detach().clamp(min=1e-6)
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
            start = int(self.lookback_len)
            end = int(self.lookback_len + h)
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

    def get_guidance_step_gate(self, step_index, time_steps=None):
        low = min(max(float(self.text_guide_step_low), 0.0), 1.0)
        high = min(max(float(self.text_guide_step_high), 0.0), 1.0)
        if high <= low:
            return 1.0
        if self.ddim and time_steps is not None:
            current_step = float(time_steps[step_index])
        else:
            current_step = float(step_index)
        tau = min(max(current_step / max(self.num_steps - 1, 1), 0.0), 1.0)
        k = max(float(self.text_guide_step_k), 1e-6)
        left = 1.0 / (1.0 + math.exp(-k * (tau - low)))
        right = 1.0 / (1.0 + math.exp(-k * (high - tau)))
        return left * right

    def impute(self, observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=None, timestep_emb=None, size_emb=None, context=None, trend_prior=None, text_mask=None):
        B, K, L = observed_data.shape
        guide_w_tensor = self._to_samplewise_weight(guide_w, B, observed_data.device)
        if text_mask is not None:
            if text_mask.dim() > 1:
                effective_guide_w = guide_w_tensor.unsqueeze(1) * text_mask.clamp(min=0.0, max=1.0).pow(self.guide_reliability_power)
            else:
                effective_guide_w = guide_w_tensor * text_mask.clamp(min=0.0, max=1.0).pow(self.guide_reliability_power)
        else:
            effective_guide_w = guide_w_tensor
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
        
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        if self.cfg:
            side_info = side_info.repeat(2, 1, 1, 1)
            if timestep_emb is not None:
                timestep_emb = timestep_emb.repeat(2, 1, 1, 1)
            if context is not None:
                context = context.repeat(2, 1, 1)
            cfg_mask = torch.zeros((2*B, )).to(self.device) 
            cfg_mask[:B] = 1.
        else:
            cfg_mask = None

        for i in range(n_samples):
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)
            for t in range(self.sample_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1) 
                else:
                    if self.decomp:
                        cond_obs = cond_mask * observed_data
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1) # (B, 1, K, L)
                        res, moving_mean = self.decomposition(cond_obs) # (B, K, L), (B, K, L)
                        res, moving_mean = res.unsqueeze(1), moving_mean.unsqueeze(1) # (B, 1, K, L), (B, 1, K, L)
                        res_input = torch.cat([res, noisy_target], dim=1)  # (B,2,K,L)
                        moving_mean_input = torch.cat([moving_mean, noisy_target], dim=1)  # (B,2,K,L)
                        if self.cfg:
                            res_input = res_input.repeat(2, 1, 1, 1) # (2*B, 2, K, L)
                            moving_mean_input = moving_mean_input.repeat(2, 1, 1, 1) # (2*B, 2, K, L)
                        seasonal_context = None if self.text_trend_only_guidance else context
                        trend_context = context
                        predicted_seasonal = self._unwrap_diffmodel_output(
                            self.diffmodel_sesonal(res_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, seasonal_context)
                        ) # (2*B, K, L)
                        predicted_trend = self._unwrap_diffmodel_output(
                            self.diffmodel_trend(moving_mean_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, trend_context)
                        ) # (2*B, K, L)
                        predicted = predicted_seasonal + predicted_trend # (2*B, K, L)
                    else:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1) # (B, 1, K, L)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1) # (B, 1, K, L)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)
                        if self.cfg:
                            diff_input = diff_input.repeat(2, 1, 1, 1) # (2*B, 2, K, L)
                        if self.save_attn:
                            predicted, attn = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, context) # (2*B, K, L)
                        else:
                            predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, context) # (2*B, K, L)
                if self.cfg:
                    predicted_cond, predicted_uncond = predicted[:B], predicted[B:]
                    guidance_step_gate = self.get_guidance_step_gate(t, time_steps if self.ddim else None)
                    effective_guide_w_t = effective_guide_w * guidance_step_gate
                    if self.trend_cfg:
                        if self.trend_cfg_random:
                            trend_prior = self.sample_random_trend_prior(B, observed_data.device)
                        if trend_prior is not None:
                            step_ratio = self.get_trend_step_ratio(t, time_steps if self.ddim else None) * guidance_step_gate
                            trend_weight = self.get_trend_guidance_weight(trend_prior, step_ratio, guide_w_tensor, text_mask)
                            seq_trend_weight = self._build_future_seq_weight(trend_weight, L)
                            if seq_trend_weight is None:
                                predicted = predicted_uncond + trend_weight[:, None, None] * (predicted_cond - predicted_uncond)
                            else:
                                predicted = predicted_uncond + seq_trend_weight * (predicted_cond - predicted_uncond)
                        else:
                            seq_weight = self._build_future_seq_weight(effective_guide_w_t, L)
                            if seq_weight is None:
                                predicted = predicted_uncond + effective_guide_w_t[:, None, None] * (predicted_cond - predicted_uncond)
                            else:
                                predicted = predicted_uncond + seq_weight * (predicted_cond - predicted_uncond)
                    else:
                        seq_weight = self._build_future_seq_weight(effective_guide_w_t, L)
                        if seq_weight is None:
                            predicted = predicted_uncond + effective_guide_w_t[:, None, None] * (predicted_cond - predicted_uncond)
                        else:
                            predicted = predicted_uncond + seq_weight * (predicted_cond - predicted_uncond)

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

            imputed_samples[:, i] = current_sample.detach()
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
            if self.text_quality_gate:
                gated_quality = text_quality_raw.clamp(min=self.text_quality_min_scale, max=1.0)
                text_mask = (text_mask > 0).float() * gated_quality
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

    def _update_reliability_debug(self, use_gate, strength_gate=None, evidence_vec=None, context_gate=None, guide_gate=None):
        if use_gate is None and strength_gate is None and context_gate is None and guide_gate is None:
            self.latest_reliability_stats = {}
            return
        use_values = None if use_gate is None else use_gate.detach().reshape(-1).float().cpu()
        strength_values = None if strength_gate is None else strength_gate.detach().reshape(-1).float().cpu()
        context_values = None if context_gate is None else context_gate.detach().reshape(-1).float().cpu()
        guide_values = None if guide_gate is None else guide_gate.detach().reshape(-1).float().cpu()
        if (
            (use_values is None or use_values.numel() == 0)
            and (strength_values is None or strength_values.numel() == 0)
            and (context_values is None or context_values.numel() == 0)
            and (guide_values is None or guide_values.numel() == 0)
        ):
            self.latest_reliability_stats = {}
            return
        stats = {}
        if use_values is not None and use_values.numel() > 0:
            stats["use_gate"] = {
                "mean": float(use_values.mean().item()),
                "std": float(use_values.std(unbiased=False).item()),
                "p10": float(torch.quantile(use_values, 0.1).item()),
                "p50": float(torch.quantile(use_values, 0.5).item()),
                "p90": float(torch.quantile(use_values, 0.9).item()),
            }
            if use_gate.dim() > 1:
                thirds = torch.chunk(use_gate.detach().float().cpu(), 3, dim=1)
                labels = ("near", "mid", "far")
                for label, chunk in zip(labels, thirds):
                    stats["use_gate"][label] = float(chunk.mean().item())
        if strength_values is not None and strength_values.numel() > 0:
            stats["strength_gate"] = {
                "mean": float(strength_values.mean().item()),
                "std": float(strength_values.std(unbiased=False).item()),
                "p10": float(torch.quantile(strength_values, 0.1).item()),
                "p50": float(torch.quantile(strength_values, 0.5).item()),
                "p90": float(torch.quantile(strength_values, 0.9).item()),
            }
            if strength_gate.dim() > 1:
                thirds = torch.chunk(strength_gate.detach().float().cpu(), 3, dim=1)
                labels = ("near", "mid", "far")
                for label, chunk in zip(labels, thirds):
                    stats["strength_gate"][label] = float(chunk.mean().item())
        for gate_name, gate, values in (
            ("context_gate", context_gate, context_values),
            ("guide_gate", guide_gate, guide_values),
        ):
            if values is not None and values.numel() > 0:
                stats[gate_name] = {
                    "mean": float(values.mean().item()),
                    "std": float(values.std(unbiased=False).item()),
                    "p10": float(torch.quantile(values, 0.1).item()),
                    "p50": float(torch.quantile(values, 0.5).item()),
                    "p90": float(torch.quantile(values, 0.9).item()),
                }
                if gate.dim() > 1:
                    thirds = torch.chunk(gate.detach().float().cpu(), 3, dim=1)
                    labels = ("near", "mid", "far")
                    for label, chunk in zip(labels, thirds):
                        stats[gate_name][label] = float(chunk.mean().item())
        if evidence_vec is not None:
            evidence_mean = evidence_vec.detach().float().mean(dim=0).cpu().tolist()
            stats["evidence_mean"] = [float(v) for v in evidence_mean]
        self.latest_reliability_stats = stats

    def _get_gate_warmup_scale(self, gate_name):
        epoch = max(int(self.current_epoch), 0)
        if gate_name == "use":
            full_epoch = max(int(self.use_gate_warmup_epochs), 0)
            if full_epoch <= 0:
                return 1.0
            return float(min(max(epoch, 0), full_epoch)) / float(full_epoch)
        delay_epoch = max(int(self.use_gate_warmup_epochs), 0)
        full_epoch = max(int(self.strength_gate_warmup_epochs), delay_epoch)
        if epoch <= delay_epoch:
            return 0.0
        if full_epoch <= delay_epoch:
            return 1.0
        return float(min(epoch - delay_epoch, full_epoch - delay_epoch)) / float(full_epoch - delay_epoch)

    def _get_gate_loss_scale(self, gate_name):
        return self._get_gate_warmup_scale(gate_name)

    def _ensure_horizon_gate(self, gate, batch_size=None):
        if gate is None:
            return None
        if gate.dim() == 1:
            return gate.unsqueeze(1).expand(-1, self.pred_len)
        if gate.dim() == 2:
            return gate
        if gate.dim() > 2:
            return gate.reshape(gate.shape[0], -1)[:, : self.pred_len]
        if batch_size is not None:
            return gate.reshape(batch_size, self.pred_len)
        return gate

    def _collapse_horizon_gate(self, gate):
        if gate is None:
            return None
        if gate.dim() <= 1:
            return gate.reshape(-1)
        return gate.mean(dim=1)

    def _resize_horizon_gate(self, gate, target_len):
        if gate is None:
            return None
        if gate.dim() <= 1:
            return gate.reshape(-1, 1).float().expand(-1, target_len)
        if gate.shape[1] == target_len:
            return gate.float()
        gate = gate.float().unsqueeze(1)
        resized = F.interpolate(gate, size=target_len, mode="linear", align_corners=False)
        return resized.squeeze(1)

    def _build_future_seq_weight(self, weight, total_len):
        if weight is None:
            return None
        if weight.dim() == 1:
            return weight[:, None, None]
        seq_weight = torch.ones((weight.shape[0], total_len), device=weight.device, dtype=weight.dtype)
        future_len = min(self.pred_len, total_len - self.lookback_len)
        if future_len > 0:
            seq_weight[:, self.lookback_len:self.lookback_len + future_len] = weight[:, :future_len]
        return seq_weight.unsqueeze(1)

    def _combine_text_evidence(self, text_evidence_vec, event_quality_summary):
        base_evidence = None if text_evidence_vec is None else text_evidence_vec.float()
        event_evidence = None if event_quality_summary is None else event_quality_summary.float()
        if event_evidence is not None and base_evidence is not None:
            return 0.4 * base_evidence + 0.6 * event_evidence
        if event_evidence is not None:
            return event_evidence
        return base_evidence

    def _compute_context_gate(self, strength_gate, evidence_vec=None):
        if strength_gate is None:
            return None
        gate = self._ensure_horizon_gate(strength_gate.float(), self.pred_len).clamp(min=0.0, max=1.0)
        if evidence_vec is None:
            ratio = torch.full_like(gate, min(max(float(self.text_context_ratio_min), 0.0), 1.0))
            dynamic_max = torch.full_like(gate, min(max(float(self.text_context_max_base), 0.0), 1.0))
        else:
            evidence = evidence_vec.float().clamp(min=0.0, max=1.0)
            if evidence.dim() == 1:
                evidence = evidence.reshape(-1, 1)
            quality = evidence[:, 0:1]
            density = evidence[:, 1:2] if evidence.shape[1] > 1 else quality
            freshness = evidence[:, 2:3] if evidence.shape[1] > 2 else quality
            novelty = evidence[:, 3:4] if evidence.shape[1] > 3 else quality
            agreement = evidence[:, 4:5] if evidence.shape[1] > 4 else quality
            regime = evidence[:, 5:6] if evidence.shape[1] > 5 else quality
            event_score = evidence[:, 6:7] if evidence.shape[1] > 6 else quality
            temporal_support = (0.5 * density + 0.5 * freshness).clamp(min=0.0, max=1.0)
            semantic_support = (0.5 * agreement + 0.5 * event_score).clamp(min=0.0, max=1.0)
            risk_adjusted = (
                quality
                * (0.35 + 0.65 * temporal_support)
                * (0.5 + 0.5 * agreement)
                * (0.6 + 0.4 * novelty)
                * (0.5 + 0.5 * event_score)
                * (0.7 + 0.3 * regime)
            ).clamp(min=0.0, max=1.0)
            ratio_min = min(max(float(self.text_context_ratio_min), 0.0), 1.0)
            ratio_max = min(max(float(self.text_context_ratio_max), ratio_min), 1.0)
            ratio = ratio_min + (ratio_max - ratio_min) * risk_adjusted
            dynamic_max = max(float(self.text_context_max_base), 0.0) + max(float(self.text_context_max_boost), 0.0) * (
                quality * temporal_support * semantic_support
            )
            ratio = ratio.expand(-1, self.pred_len)
            dynamic_max = dynamic_max.expand(-1, self.pred_len).clamp(min=0.0, max=1.0)
        if self.text_context_horizon_bias != 0.0:
            horizon_pos = torch.linspace(0.0, 1.0, self.pred_len, device=gate.device, dtype=gate.dtype).view(1, -1)
            horizon_bias = (1.0 + self.text_context_horizon_bias * (0.5 - horizon_pos)).clamp(min=0.0)
            ratio = ratio * horizon_bias
        context_gate = gate * ratio
        context_gate = torch.minimum(context_gate, dynamic_max)
        return context_gate.clamp(min=0.0, max=1.0)

    def _compute_guide_gate(self, strength_gate, evidence_vec=None, trend_align=None):
        if strength_gate is None:
            return None
        gate = self._ensure_horizon_gate(strength_gate.float(), self.pred_len).clamp(min=0.0, max=1.0)
        if evidence_vec is None:
            ratio = torch.full_like(gate, min(max(float(self.text_guide_ratio_max), 0.0), 1.0))
            dynamic_max = torch.full_like(gate, min(max(float(self.text_guide_max_base), 0.0), 1.0))
        else:
            evidence = evidence_vec.float().clamp(min=0.0, max=1.0)
            if evidence.dim() == 1:
                evidence = evidence.reshape(-1, 1)
            quality = evidence[:, 0:1]
            density = evidence[:, 1:2] if evidence.shape[1] > 1 else quality
            freshness = evidence[:, 2:3] if evidence.shape[1] > 2 else quality
            agreement = evidence[:, 4:5] if evidence.shape[1] > 4 else quality
            event_score = evidence[:, 6:7] if evidence.shape[1] > 6 else quality
            source_support = torch.sqrt((agreement * event_score).clamp(min=0.0, max=1.0))
            temporal_support = ((0.25 + 0.75 * freshness) * (0.25 + 0.75 * density)).clamp(min=0.0, max=1.0)
            strict_quality = (quality * source_support * temporal_support).clamp(min=0.0, max=1.0)
            quality_power = max(float(self.text_guide_quality_power), 1e-6)
            ratio_max = min(max(float(self.text_guide_ratio_max), 0.0), 1.0)
            ratio = ratio_max * strict_quality.pow(quality_power)
            dynamic_max = max(float(self.text_guide_max_base), 0.0) + max(float(self.text_guide_max_boost), 0.0) * (
                quality * source_support * freshness
            )
            ratio = ratio.expand(-1, self.pred_len)
            dynamic_max = dynamic_max.expand(-1, self.pred_len).clamp(min=0.0, max=1.0)
        if trend_align is not None:
            align = self._ensure_horizon_gate(trend_align.float(), self.pred_len).clamp(min=0.0, max=1.0)
            ratio = ratio * align
            dynamic_max = dynamic_max * (0.25 + 0.75 * align)
        guide_gate = gate * ratio
        guide_gate = torch.minimum(guide_gate, dynamic_max)
        return guide_gate.clamp(min=0.0, max=1.0)

    def _compute_horizon_quality_prior(self, text_event_quality_feats, text_event_time_deltas, text_event_mask, fallback_quality):
        fallback = fallback_quality.reshape(-1, 1).float().clamp(0.0, 1.0).expand(-1, self.pred_len)
        if text_event_quality_feats is None or text_event_time_deltas is None or text_event_mask is None:
            return fallback
        quality_feats = text_event_quality_feats.float().clamp(min=0.0, max=1.0)
        event_mask = text_event_mask.float().clamp(min=0.0, max=1.0)
        event_strength = quality_feats.mean(dim=-1) * event_mask
        horizon_index = torch.arange(self.pred_len, device=quality_feats.device, dtype=quality_feats.dtype).view(1, 1, -1)
        event_time = text_event_time_deltas.float().unsqueeze(-1)
        relevance = torch.exp(-torch.abs(event_time - horizon_index) / max(self.text_recency_tau_days, 1e-6))
        weighted = event_strength.unsqueeze(-1) * relevance
        denom = (relevance * event_mask.unsqueeze(-1)).sum(dim=1).clamp(min=1e-6)
        horizon_prior = weighted.sum(dim=1) / denom
        return torch.maximum(horizon_prior.clamp(0.0, 1.0), fallback)

    def _build_horizon_trend_priors(self, trend_prior_num, trend_prior_text, semantic_state, text_mask=None):
        if trend_prior_num is None and trend_prior_text is None:
            return None, None
        if trend_prior_num is None:
            trend_prior_num = trend_prior_text
        if trend_prior_text is None:
            trend_prior_text = trend_prior_num
        num_prior = trend_prior_num.float().unsqueeze(1).expand(-1, self.pred_len, -1)
        text_prior = trend_prior_text.float().unsqueeze(1).expand(-1, self.pred_len, -1)
        if semantic_state is not None and hasattr(self, "semantic_to_trend"):
            semantic_delta = self.semantic_to_trend(semantic_state.float()).reshape(-1, self.pred_len, 3)
            semantic_delta = torch.tanh(semantic_delta)
            semantic_delta = torch.cat(
                [
                    semantic_delta[:, :, :1],
                    0.5 * semantic_delta[:, :, 1:],
                ],
                dim=-1,
            )
            text_prior = text_prior + self.semantic_trend_scale * semantic_delta
            text_prior = torch.cat(
                [
                    text_prior[:, :, :1],
                    text_prior[:, :, 1:].clamp(min=0.0),
                ],
                dim=-1,
            )
        diff = torch.abs(text_prior - num_prior).sum(dim=-1)
        align = torch.exp(-self.text_numeric_align_gamma * diff).clamp(0.0, 1.0)
        if text_mask is not None:
            align = align * self._ensure_horizon_gate(text_mask).float().clamp(0.0, 1.0)
        fused = align.unsqueeze(-1) * text_prior + (1.0 - align).unsqueeze(-1) * num_prior
        return fused, align

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

    def _encode_event_texts(self, text_event_texts, text_event_mask):
        if not self.with_texts or text_event_texts is None or text_event_mask is None:
            return None
        batch_size = len(text_event_texts)
        if batch_size <= 0:
            return None
        event_count = len(text_event_texts[0]) if len(text_event_texts[0]) > 0 else 0
        if event_count <= 0:
            return None
        flat_texts = []
        for row in text_event_texts:
            if len(row) != event_count:
                row = list(row) + ["NA"] * max(event_count - len(row), 0)
                row = row[:event_count]
            flat_texts.extend(row)
        _, pooled, _ = self._encode_text_source_with_options(
            flat_texts,
            max_length=self.event_text_max_length,
            return_sequence=False,
        )
        pooled = pooled.reshape(batch_size, event_count, -1)
        if text_event_mask is not None:
            pooled = pooled * text_event_mask.unsqueeze(-1).float()
        return pooled

    def _compute_event_quality_scores(self, text_event_quality_feats, text_event_mask):
        if text_event_quality_feats is None or text_event_mask is None:
            return None
        feats = text_event_quality_feats.float().clamp(min=1e-4, max=1.0)
        log_score = torch.log(feats).mean(dim=-1)
        log_score = log_score + torch.log(text_event_mask.float().clamp(min=1e-4))
        masked_log_score = log_score.masked_fill(text_event_mask <= 0, float("-inf"))
        all_invalid = (text_event_mask.sum(dim=1) <= 0)
        if all_invalid.any():
            masked_log_score = masked_log_score.clone()
            masked_log_score[all_invalid] = 0.0
        scores = torch.softmax(self.event_quality_beta * masked_log_score, dim=1)
        scores = scores * text_event_mask.float()
        denom = scores.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return scores / denom

    def _summarize_event_quality(self, text_event_quality_feats, event_scores, text_event_mask, base_evidence=None):
        if text_event_quality_feats is None or event_scores is None or text_event_mask is None:
            return base_evidence
        feats = text_event_quality_feats.float()
        mask = text_event_mask.float()
        weights = event_scores * mask
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weighted = (weights.unsqueeze(-1) * feats).sum(dim=1) / denom
        density = (mask.sum(dim=1) / float(mask.shape[1])).clamp(0.0, 1.0)
        mean_score = (weights * feats.mean(dim=-1)).sum(dim=1) / denom.squeeze(1)
        regime = torch.zeros_like(mean_score)
        event_strength = weighted[:, 5]
        if base_evidence is not None and base_evidence.shape[1] >= 7:
            regime = base_evidence[:, 5].float().clamp(0.0, 1.0)
            event_strength = 0.5 * event_strength + 0.5 * base_evidence[:, 6].float().clamp(0.0, 1.0)
        return torch.stack(
            [
                mean_score.clamp(0.0, 1.0),
                density,
                weighted[:, 1].clamp(0.0, 1.0),
                weighted[:, 3].clamp(0.0, 1.0),
                weighted[:, 4].clamp(0.0, 1.0),
                regime,
                event_strength.clamp(0.0, 1.0),
            ],
            dim=1,
        )

    def _build_event_semantic_state(self, text_event_texts, text_event_source_ids, text_event_quality_feats, text_event_mask, base_evidence=None):
        if (
            not self.with_texts
            or text_event_texts is None
            or text_event_source_ids is None
            or text_event_quality_feats is None
            or text_event_mask is None
        ):
            return None, base_evidence
        event_pooled = self._encode_event_texts(text_event_texts, text_event_mask)
        if event_pooled is None:
            return None, base_evidence
        source_emb = self.event_source_embed(text_event_source_ids.long().clamp(min=0, max=2))
        event_features = torch.cat(
            [
                event_pooled.float(),
                source_emb.float(),
                text_event_quality_feats.float(),
            ],
            dim=-1,
        )
        event_semantics = self.event_semantic_proj(event_features)
        event_scores = self._compute_event_quality_scores(text_event_quality_feats, text_event_mask)
        if event_scores is None:
            return None, base_evidence
        semantic_state = (event_scores.unsqueeze(-1) * event_semantics).sum(dim=1)
        event_summary = self._summarize_event_quality(
            text_event_quality_feats,
            event_scores,
            text_event_mask,
            base_evidence=base_evidence,
        )
        return semantic_state, event_summary

    def _compute_text_gates(self, text_evidence_vec, text_mask=None, text_pooled=None, trend_prior_num=None, trend_prior_text=None, event_semantic_state=None, event_quality_summary=None, text_event_quality_feats=None, text_event_time_deltas=None, text_event_mask=None):
        if (text_evidence_vec is None and event_quality_summary is None) or not self.with_texts:
            use_gate = text_mask
            strength_gate = text_mask
            self._update_reliability_debug(use_gate, strength_gate, None)
            return use_gate, strength_gate, None
        evidence = self._combine_text_evidence(text_evidence_vec, event_quality_summary)
        if text_mask is not None:
            text_mask = text_mask.reshape(-1).float().clamp(0.0, 1.0)
        base_quality = evidence[:, 0].clamp(0.0, 1.0)
        if text_mask is not None:
            base_quality = base_quality * text_mask
        horizon_quality_prior = self._compute_horizon_quality_prior(
            text_event_quality_feats,
            text_event_time_deltas,
            text_event_mask,
            base_quality,
        )
        if text_mask is not None:
            horizon_quality_prior = horizon_quality_prior * text_mask.unsqueeze(1)

        semantic_state = event_semantic_state
        if semantic_state is None and text_pooled is not None:
            semantic_state = self.semantic_proj(text_pooled.float())

        raw_use_gate = torch.sigmoid(self.use_gate_head(evidence))
        raw_use_gate = self.reliability_min + (1.0 - self.reliability_min) * raw_use_gate
        use_scale = self._get_gate_warmup_scale("use")
        use_gate = (1.0 - use_scale) * horizon_quality_prior + use_scale * raw_use_gate
        use_gate = self.use_gate_min + (1.0 - self.use_gate_min) * use_gate
        if text_mask is not None:
            use_gate = use_gate * text_mask.unsqueeze(1)

        if trend_prior_num is None:
            trend_prior_num = torch.zeros((evidence.shape[0], 3), device=evidence.device)
        if trend_prior_text is None:
            trend_prior_text = trend_prior_num
        if semantic_state is None:
            semantic_state = torch.zeros((evidence.shape[0], self.semantic_dim), device=evidence.device)

        strength_features = torch.cat(
            [
                evidence,
                semantic_state,
                trend_prior_num.float(),
                trend_prior_text.float(),
            ],
            dim=1,
        )
        raw_strength_gate = torch.sigmoid(self.strength_gate_head(strength_features))
        raw_strength_gate = self.reliability_min + (1.0 - self.reliability_min) * raw_strength_gate
        strength_scale = self._get_gate_warmup_scale("strength")
        strength_gate = (1.0 - strength_scale) * horizon_quality_prior + strength_scale * raw_strength_gate
        use_floor = min(max(self.strength_use_mix_floor, 0.0), 1.0)
        strength_gate = strength_gate * (use_floor + (1.0 - use_floor) * use_gate)
        if self.horizon_strength_bias != 0.0:
            horizon_pos = torch.linspace(0.0, 1.0, self.pred_len, device=strength_gate.device, dtype=strength_gate.dtype).view(1, -1)
            bias = (1.0 + self.horizon_strength_bias * (0.5 - horizon_pos)).clamp(min=0.0)
            strength_gate = strength_gate * bias
        strength_gate = self.strength_gate_min + (1.0 - self.strength_gate_min) * strength_gate
        if text_mask is not None:
            strength_gate = strength_gate * text_mask.unsqueeze(1)
        strength_gate = strength_gate.clamp(min=0.0, max=1.0)

        self._update_reliability_debug(use_gate, strength_gate, evidence)
        return use_gate, strength_gate, semantic_state

    def get_trend_guidance_weight(self, trend_prior, step_ratio, guide_w, text_mask=None):
        base_weight = self._to_samplewise_weight(guide_w, trend_prior.shape[0], trend_prior.device)
        if trend_prior.dim() == 3:
            strength = trend_prior[:, :, 1].clamp(min=0.0)
            volatility = trend_prior[:, :, 2].clamp(min=0.0) * self.trend_volatility_scale
            base_weight = base_weight.unsqueeze(1).expand(-1, trend_prior.shape[1])
        else:
            strength = trend_prior[:, 1].clamp(min=0.0)
            volatility = trend_prior[:, 2].clamp(min=0.0) * self.trend_volatility_scale
        strength = 1.0 + self.trend_strength_scale * (strength - 1.0)
        strength = strength.clamp(min=0.0)
        vol_penalty = 1.0 / (1.0 + volatility)
        weight = base_weight * step_ratio * strength * vol_penalty
        if text_mask is not None:
            weight = weight * text_mask
        return weight

    def sample_random_trend_prior(self, batch_size, device):
        direction = torch.randint(0, 3, (batch_size,), device=device).float() - 1.0
        strength_choices = torch.tensor([0.5, 1.0, 1.5], device=device)
        volatility_choices = torch.tensor([0.0, 0.5, 1.0], device=device)
        strength = strength_choices[torch.randint(0, 3, (batch_size,), device=device)]
        volatility = volatility_choices[torch.randint(0, 3, (batch_size,), device=device)]
        return torch.stack([direction, strength, volatility], dim=1)

    def fuse_trend_priors(self, trend_prior_num, trend_prior_text, text_mask=None):
        if trend_prior_num is None and trend_prior_text is None:
            return None, None
        if trend_prior_num is None:
            trend_prior_num = trend_prior_text
        if trend_prior_text is None:
            trend_prior_text = trend_prior_num
        diff = torch.abs(trend_prior_text - trend_prior_num).sum(dim=1)
        align = torch.exp(-self.text_numeric_align_gamma * diff).clamp(0.0, 1.0)
        if text_mask is not None:
            if text_mask.dim() > 1:
                align = align.unsqueeze(1) * text_mask.float().clamp(0.0, 1.0)
                fused = align.unsqueeze(-1) * trend_prior_text.unsqueeze(1) + (1.0 - align).unsqueeze(-1) * trend_prior_num.unsqueeze(1)
                return fused, align
            align = align * text_mask.reshape(-1).float().clamp(0.0, 1.0)
        fused = align.unsqueeze(1) * trend_prior_text + (1.0 - align).unsqueeze(1) * trend_prior_num
        return fused, align
    
    def _encode_text_source(self, text):
        return self._encode_text_source_with_options(text, max_length=self.text_max_length, return_sequence=True)

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

    def _build_text_context(self, encoded_text, text_scale):
        if encoded_text is None or text_scale is None:
            return None
        token_scale = self._resize_horizon_gate(text_scale, encoded_text.shape[1]).clamp(min=0.0, max=1.0)
        context = encoded_text * token_scale.unsqueeze(-1)
        return context.permute(0, 2, 1)

    def _summarize_numeric_context(self, observed_data, cond_mask):
        lookback = observed_data[:, :, :self.lookback_len]
        lookback_mask = cond_mask[:, :, :self.lookback_len]
        denom = lookback_mask.sum(dim=2).clamp(min=1.0)
        mean = (lookback * lookback_mask).sum(dim=2) / denom
        centered = (lookback - mean.unsqueeze(2)) * lookback_mask
        std = torch.sqrt((centered ** 2).sum(dim=2) / denom + 1e-6)
        first = lookback[:, :, 0]
        last = lookback[:, :, self.lookback_len - 1]
        slope = last - first
        mean_abs = (lookback.abs() * lookback_mask).sum(dim=2) / denom
        return torch.stack(
            [
                mean.mean(dim=1),
                std.mean(dim=1),
                slope.mean(dim=1),
                mean_abs.mean(dim=1),
            ],
            dim=1,
        )

    def _compute_text_benefit_gate(self, observed_data, cond_mask, text_pooled, text_mask, trend_align, domain_coverage=None):
        if text_pooled is None or text_mask is None:
            return None
        numeric_summary = self._summarize_numeric_context(observed_data, cond_mask)
        quality = text_mask.reshape(-1, 1).float().clamp(0.0, 1.0)
        if trend_align is None:
            trend_align = torch.ones_like(quality.squeeze(1))
        align = trend_align.reshape(-1, 1).float().clamp(0.0, 1.0)
        if domain_coverage is None:
            domain_coverage = torch.ones(text_pooled.shape[0], device=text_pooled.device)
        cov = domain_coverage.reshape(-1, 1).float().clamp(0.0, 1.0)
        benefit_input = torch.cat([numeric_summary, text_pooled, quality, align, cov], dim=1)
        benefit_gate = torch.sigmoid(self.text_benefit_head(benefit_input)).reshape(-1)
        coverage_cap = domain_coverage.reshape(-1).float().clamp(0.0, 1.0).pow(self.coverage_power)
        benefit_gate = benefit_gate * coverage_cap
        return benefit_gate

    def _get_aug_inputs(self, batch):
        text_ret = batch.get("text_ret", batch.get("retrieved_text"))
        text_cot = batch.get("text_cot", batch.get("cot_text"))
        quality_ret = batch.get("text_quality_ret")
        quality_cot = batch.get("text_quality_cot")
        if quality_ret is not None:
            quality_ret = quality_ret.to(self.device).float().reshape(-1)
        if quality_cot is not None:
            quality_cot = quality_cot.to(self.device).float().reshape(-1)
        return text_ret, text_cot, quality_ret, quality_cot

    def _build_augmented_context(
        self,
        batch,
        context_raw,
        raw_gate,
        trend_prior_num,
        trend_prior_text,
        trend_align,
        raw_token_len,
    ):
        if context_raw is None or raw_gate is None or raw_token_len is None:
            return context_raw, None
        text_ret, text_cot, quality_ret, quality_cot = self._get_aug_inputs(batch)
        if text_ret is None or text_cot is None or quality_ret is None or quality_cot is None:
            return context_raw, None
        _, pooled_ret, _ = self._encode_text_source(text_ret)
        _, pooled_cot, _ = self._encode_text_source(text_cot)
        if trend_prior_num is None:
            trend_prior_num = torch.zeros((pooled_ret.shape[0], 3), device=self.device)
        if trend_prior_text is None:
            trend_prior_text = trend_prior_num
        if trend_align is None:
            trend_align = torch.ones((pooled_ret.shape[0],), device=self.device)
        if trend_align.dim() > 1:
            align = trend_align.mean(dim=1, keepdim=True).float().clamp(0.0, 1.0)
        else:
            align = trend_align.reshape(-1, 1).float().clamp(0.0, 1.0)
        q_ret = quality_ret.reshape(-1, 1).float().clamp(0.0, 1.0)
        q_cot = quality_cot.reshape(-1, 1).float().clamp(0.0, 1.0)
        aug_features = torch.cat(
            [
                pooled_ret,
                pooled_cot,
                trend_prior_text.float(),
                trend_prior_num.float(),
                q_ret,
                q_cot,
                align,
            ],
            dim=1,
        )
        aug_base_gate = torch.sigmoid(self.text_aug_gate_head(aug_features)).reshape(-1)
        aug_available = torch.maximum(q_ret.reshape(-1), q_cot.reshape(-1))
        token_gate = self._resize_horizon_gate(raw_gate, raw_token_len).clamp(min=0.0, max=1.0)
        aug_gate = self.text_aug_max_ratio * token_gate * aug_base_gate.unsqueeze(1) * aug_available.unsqueeze(1)
        if torch.all(aug_gate <= 0):
            return context_raw, aug_gate
        aug_residual = self.text_aug_adapter(aug_features)
        aug_residual = aug_residual.unsqueeze(2).expand(-1, self.text_hidden_dim, raw_token_len)
        context_aug = context_raw + aug_gate.unsqueeze(1) * aug_residual
        return context_aug, aug_gate

    def get_text_info(self, text, text_mask):
        encoded_text, _, token_input = self._encode_text_source(text)
        context = self._build_text_context(encoded_text, text_mask)
        if self.save_token:
            tokens_str = self.tokenizer.batch_decode(token_input['input_ids'])
            return context, tokens_str
        else:
            return context

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
            encoded_text, text_pooled, token_input = self._encode_text_source(texts)
            event_semantic_state, event_quality_summary = self._build_event_semantic_state(
                text_event_texts,
                text_event_source_ids,
                text_event_quality_feats,
                text_event_mask,
                base_evidence=text_evidence_vec,
            )
        else:
            encoded_text = None
            text_pooled = None
            event_semantic_state = None
            event_quality_summary = text_evidence_vec
        use_gate, strength_gate, semantic_state = self._compute_text_gates(
            text_evidence_vec,
            text_mask=text_mask,
            text_pooled=text_pooled,
            trend_prior_num=trend_prior_num,
            trend_prior_text=trend_prior_text,
            event_semantic_state=event_semantic_state,
            event_quality_summary=event_quality_summary,
            text_event_quality_feats=text_event_quality_feats,
            text_event_time_deltas=text_event_time_deltas,
            text_event_mask=text_event_mask,
        )
        trend_prior_eff, trend_align = self._build_horizon_trend_priors(
            trend_prior_num,
            trend_prior_text,
            semantic_state,
            text_mask=use_gate,
        )
        context_evidence = self._combine_text_evidence(text_evidence_vec, event_quality_summary)
        context_gate = self._compute_context_gate(strength_gate, context_evidence)
        guide_gate = self._compute_guide_gate(strength_gate, context_evidence, trend_align)
        self._update_reliability_debug(use_gate, strength_gate, text_evidence_vec, context_gate=context_gate, guide_gate=guide_gate)
        context_raw = self._build_text_context(encoded_text, context_gate) if encoded_text is not None else None
        raw_token_len = encoded_text.shape[1] if encoded_text is not None else None
        context, aug_gate = self._build_augmented_context(
            batch,
            context_raw,
            context_gate,
            trend_prior_num,
            trend_prior_text,
            trend_align,
            raw_token_len,
        )
        semantic_text_mask = guide_gate
        effective_text_mask = semantic_text_mask
        trend_prior_for_multires = trend_prior_num
        if self.multi_res_trend_source in {"text_fused", "fused", "text"} and trend_prior_eff is not None:
            trend_prior_for_multires = trend_prior_eff

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
            context=context,
            trend_prior=trend_prior_for_multires,
            text_mask=effective_text_mask,
            use_gate=use_gate,
            context_raw=context_raw,
            aug_gate=aug_gate,
            strength_gate=strength_gate,
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

            if self.with_texts:
                encoded_text, text_pooled, token_input = self._encode_text_source(texts)
                event_semantic_state, event_quality_summary = self._build_event_semantic_state(
                    text_event_texts,
                    text_event_source_ids,
                    text_event_quality_feats,
                    text_event_mask,
                    base_evidence=text_evidence_vec,
                )
                use_gate, strength_gate, semantic_state = self._compute_text_gates(
                    text_evidence_vec,
                    text_mask=text_mask,
                    text_pooled=text_pooled,
                    trend_prior_num=trend_prior_num,
                    trend_prior_text=trend_prior_text,
                    event_semantic_state=event_semantic_state,
                    event_quality_summary=event_quality_summary,
                    text_event_quality_feats=text_event_quality_feats,
                    text_event_time_deltas=text_event_time_deltas,
                    text_event_mask=text_event_mask,
                )
                trend_prior_eff, trend_align = self._build_horizon_trend_priors(
                    trend_prior_num,
                    trend_prior_text,
                    semantic_state,
                    text_mask=use_gate,
                )
                context_evidence = self._combine_text_evidence(text_evidence_vec, event_quality_summary)
                context_gate = self._compute_context_gate(strength_gate, context_evidence)
                guide_gate = self._compute_guide_gate(strength_gate, context_evidence, trend_align)
                self._update_reliability_debug(use_gate, strength_gate, text_evidence_vec, context_gate=context_gate, guide_gate=guide_gate)
                context_raw = self._build_text_context(encoded_text, context_gate)
                raw_token_len = encoded_text.shape[1]
                context, aug_gate = self._build_augmented_context(
                    batch,
                    context_raw,
                    context_gate,
                    trend_prior_num,
                    trend_prior_text,
                    trend_align,
                    raw_token_len,
                )
                semantic_text_mask = guide_gate
                if self.save_token:
                    tokens = self.tokenizer.batch_decode(token_input['input_ids'])
            else:
                event_semantic_state, event_quality_summary = self._build_event_semantic_state(
                    text_event_texts,
                    text_event_source_ids,
                    text_event_quality_feats,
                    text_event_mask,
                    base_evidence=text_evidence_vec,
                )
                use_gate, strength_gate, semantic_state = self._compute_text_gates(
                    text_evidence_vec,
                    text_mask=text_mask,
                    text_pooled=None,
                    trend_prior_num=trend_prior_num,
                    trend_prior_text=trend_prior_text,
                    event_semantic_state=event_semantic_state,
                    event_quality_summary=event_quality_summary,
                    text_event_quality_feats=text_event_quality_feats,
                    text_event_time_deltas=text_event_time_deltas,
                    text_event_mask=text_event_mask,
                )
                trend_prior_eff, trend_align = self._build_horizon_trend_priors(
                    trend_prior_num,
                    trend_prior_text,
                    semantic_state,
                    text_mask=use_gate,
                )
                context = None
                aug_gate = None
                context_raw = None
                context_evidence = self._combine_text_evidence(text_evidence_vec, event_quality_summary)
                context_gate = self._compute_context_gate(strength_gate, context_evidence)
                guide_gate = self._compute_guide_gate(strength_gate, context_evidence, trend_align)
                self._update_reliability_debug(use_gate, strength_gate, text_evidence_vec, context_gate=context_gate, guide_gate=guide_gate)
                semantic_text_mask = guide_gate
            text_mask_f = text_mask.float() if text_mask is not None else None
            if semantic_text_mask is not None:
                text_mask_f = semantic_text_mask.float()
            if text_mask_f is not None and trend_align is not None and semantic_text_mask is None:
                text_mask_f = text_mask_f * trend_align.float()
            if self.save_attn:
                samples, attn = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=context, trend_prior=trend_prior_eff, text_mask=text_mask_f)
            else:
                samples = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=context, trend_prior=trend_prior_eff, text_mask=text_mask_f)

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
