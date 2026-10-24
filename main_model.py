import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from diff_models import diff_CSDI
from utils.prepare4llm import get_llm
from utils.controller import ForecastPolicyController
from utils.forecast_modules import CoarseForecastHead, UncertaintyHead
from utils.text_gate import TextValidityGate, benefit_label

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
    
class DLinearBaseline(nn.Module):
    """Shared-weight DLinear baseline (Zeng et al., 2023). One decomposition, two maps L→H."""

    def __init__(self, seq_len, pred_len, kernel_size=25):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size > int(seq_len):
            kernel_size = int(seq_len) if int(seq_len) % 2 == 1 else max(int(seq_len) - 1, 1)
        kernel_size = max(kernel_size, 1)
        self.decomposition = series_decomp(kernel_size)
        self.linear_seasonal = nn.Linear(int(seq_len), int(pred_len))
        self.linear_trend = nn.Linear(int(seq_len), int(pred_len))

    def forward(self, history):
        seasonal, trend = self.decomposition(history.permute(0, 2, 1))
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1))
        trend = self.linear_trend(trend.permute(0, 2, 1))
        return seasonal + trend


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
    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "text_encoder_eval", False) and getattr(self, "text_encoder", None) is not None:
            self.text_encoder.eval()
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._text_encode_cache = {}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

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
        # Off for legacy runs. xattn configs turn it on. It corrects the diffusion
        # index passed to the denoiser, the x0 DDIM direction scale, and the x0 DDPM endpoint.
        self.sampler_fix = bool(config["diffusion"].get("sampler_fix", False))
        # Legacy leaves the frozen text encoder in whatever mode model.train() sets.
        # xattn keeps it in eval so its dropout cannot change a repeated string.
        self.text_encoder_eval = bool(config["model"].get("text_encoder_eval", False))
        self.text_encoder = None
        # Memo of eval-mode encodings. Keyed only by the string and the truncation
        # settings, so a value does not depend on the split or on other samples.
        # load_state_dict clears it.
        self._text_encode_cache = {}
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
        self.multi_res_difficulty_gamma_configured = "multi_res_difficulty_gamma" in train_cfg
        self.multi_res_difficulty_gamma = float(train_cfg.get("multi_res_difficulty_gamma", 0.5))
        self.auxiliary_loss_max_ratio = float(train_cfg.get("auxiliary_loss_max_ratio", 0.0))
        self.current_epoch = 0
        self.total_epochs = max(int(train_cfg.get("epochs", 1)), 1)
        self.pattern_residual_diffusion = bool(config["model"].get("pattern_residual_diffusion", True))
        self.pattern_text_evidence = bool(config["model"].get("pattern_text_evidence", True))
        self.pattern_hidden_dim = int(config["model"].get("pattern_hidden_dim", 64))
        self.pattern_stats_dim = 8
        self.pattern_expert_set = str(config["model"].get("pattern_expert_set", "legacy5")).lower()
        self.pattern_num_experts = 4 if self.pattern_expert_set == "fpp4" else 5
        self.pattern_router_mode = str(config["model"].get("pattern_router_mode", "legacy")).lower()
        self.baseline_form = str(config["model"].get("baseline_form", "router")).lower()
        self.text_injection = str(config["model"].get("text_injection", "legacy")).lower()
        self.timestep_outer_mix = str(config["model"].get("timestep_outer_mix", "legacy")).lower()
        self.n_text_sources = int(config["model"].get("n_text_sources", 2))
        self.cot_as_source = bool(config["model"].get("cot_as_source", False))
        self.text_validity_gate = bool(config["model"].get("text_validity_gate", False))
        self.text_gate_mode = str(config["model"].get("text_gate_mode", "learned")).lower()
        self.text_validity_loss = bool(config["model"].get("text_validity_loss", self.text_validity_gate))
        self.cfg_output_combine = bool(config["model"].get("cfg_output_combine", False))
        self.shuffle_text_in_batch = bool(config["model"].get("shuffle_text_in_batch", False))
        self.season_period = int(config["model"].get("season_period", 12))
        self.benefit_eps = float(config["model"].get("benefit_eps", 1e-8))
        self.mask_noisy_target = bool(config["diffusion"].get("mask_noisy_target", False))
        self.valid_loss_main_only = bool(train_cfg.get("valid_loss_main_only", False))
        if self.baseline_form == "none":
            self.pattern_residual_diffusion = False
        self.pattern_consistency_weight = float(train_cfg.get("pattern_consistency_weight", 0.03))
        self.pattern_expert_weight = float(train_cfg.get("pattern_expert_weight", 0.05))
        self.pattern_baseline_detach = bool(train_cfg.get("pattern_baseline_detach", False))
        self.pattern_router_temperature = float(config["model"].get("pattern_router_temperature", 1.0))
        self.guide_w_default = float(config["model"].get("guide_w_default", 1.0))
        self.guide_mode = str(config["model"].get("guide_mode", "manual")).lower()
        self.final_method = bool(config["model"].get("final_method", self.guide_mode == "auto"))
        self.target_feature_index = int(config["model"].get("target_feature_index", 0))
        controller_cfg = dict(config["model"].get("forecast_policy_controller", {}) or {})
        self.use_forecast_policy_controller = bool(
            config["model"].get("use_forecast_policy_controller", controller_cfg.get("enabled", False))
        )
        if self.use_forecast_policy_controller:
            controller_cfg["enabled"] = True
            self.forecast_policy_controller = ForecastPolicyController(self.pred_len, controller_cfg)
        else:
            self.forecast_policy_controller = None
        self.policy_cond_dim = int(config["model"].get("policy_cond_dim", 16))
        self.use_aux_forecast_heads = bool(config["model"].get("use_aux_forecast_heads", self.final_method))
        self.use_coarse_forecast_head = bool(config["model"].get("use_coarse_forecast_head", self.use_aux_forecast_heads))
        self.use_uncertainty_head = bool(config["model"].get("use_uncertainty_head", self.use_aux_forecast_heads))
        self.coarse_forecast_weight = float(train_cfg.get("coarse_forecast_weight", 0.05 if self.use_coarse_forecast_head else 0.0))
        self.uncertainty_forecast_weight = float(train_cfg.get("uncertainty_forecast_weight", 0.02 if self.use_uncertainty_head else 0.0))
        self.coarse_forecast_blend = float(config["model"].get("coarse_forecast_blend", 0.15))
        self.coarse_forecast_factor = int(config["model"].get("coarse_forecast_factor", 4))
        self.aux_forecast_hidden_dim = int(config["model"].get("aux_forecast_hidden_dim", 128))
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
        segment_reliability_tensor = torch.ones(reliability_size, dtype=torch.float32)
        segment_reliabilities = train_cfg.get("multi_res_segment_reliabilities", None)
        resolved_segment_reliabilities = self._resolve_multi_res_horizon_reliabilities(segment_reliabilities)
        if len(resolved_segment_reliabilities) == len(self.multi_res_horizons):
            segment_reliability_tensor = torch.tensor(resolved_segment_reliabilities, dtype=torch.float32)
        self.multi_res_segment_reliabilities = resolved_segment_reliabilities or list(self.multi_res_horizon_reliabilities)
        self.register_buffer("multi_res_segment_reliability", segment_reliability_tensor)
        self.multi_res_huber_deltas = self._resolve_multi_res_huber_deltas(
            self.multi_res_horizons, self.multi_res_huber_deltas_cfg
        )
        self.register_buffer("pattern_q_b_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("pattern_q_b_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("pattern_segment_q_b_sum", torch.zeros(reliability_size, dtype=torch.float32))
        self.register_buffer("pattern_segment_q_b_count", torch.zeros(reliability_size, dtype=torch.float32))
        self.register_buffer("diag_segment_alpha_sum", torch.zeros(reliability_size, dtype=torch.float32))
        self.register_buffer("diag_segment_alpha_sq_sum", torch.zeros(reliability_size, dtype=torch.float32))
        self.register_buffer("diag_segment_alpha_count", torch.zeros(reliability_size, dtype=torch.float32))
        self.register_buffer("diag_segment_baseline_mse_sum", torch.zeros(reliability_size, dtype=torch.float32))
        self.register_buffer("diag_segment_baseline_mse_count", torch.zeros(reliability_size, dtype=torch.float32))
        self.register_buffer("text_gain_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("text_gain_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_text_gain_batch_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_text_gain_batch_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_w_auto_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_w_auto_sq_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_w_auto_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_text_quality_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_text_quality_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_router_agreement_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_router_agreement_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_alpha_b_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_alpha_b_sq_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_alpha_b_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_pattern_baseline_mse_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_pattern_baseline_mse_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_main_loss_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_pattern_aux_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_multi_res_aux_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_coarse_aux_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_uncertainty_aux_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_pattern_budget_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_pattern_budget_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_mr_budget_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_mr_budget_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("diag_loss_count", torch.zeros((), dtype=torch.float32))

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
            if str(self.llm).lower() == "hash":
                self.text_encoder = None
                self.tokenizer = None
                self.text_hidden_dim = int(self.context_dim)
            else:
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
            if self.text_encoder_eval and self.text_encoder is not None:
                self.text_encoder.eval()

        self.pattern_router = None
        self.pattern_evidence_head = None
        self.pattern_text_head = None
        self.pattern_router_linear = None
        self.dlinear_baseline = None
        self.text_gate = None
        if self.pattern_router_mode == "legacy":
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
        elif (
            self.pattern_router_mode in ("linear_history", "linear_history_text")
            and self.baseline_form not in ("naive", "dlinear", "none")
        ):
            self.pattern_router_linear = nn.Linear(self.lookback_len, self.pattern_num_experts)
            if self.pattern_router_mode == "linear_history_text" and self.with_texts:
                self.pattern_text_head = nn.Sequential(
                    nn.Linear(self.text_hidden_dim, self.pattern_hidden_dim),
                    nn.LayerNorm(self.pattern_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.pattern_hidden_dim, self.pattern_num_experts),
                )
        if self.baseline_form == "dlinear":
            self.dlinear_baseline = DLinearBaseline(self.lookback_len, self.pred_len)
        if self.text_validity_gate and self.text_gate_mode == "learned":
            self.text_gate = TextValidityGate(self.n_text_sources)
        self.register_buffer("gate_label_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("gate_label_count", torch.zeros((), dtype=torch.float32))
        if self.use_coarse_forecast_head:
            self.coarse_forecast_head = CoarseForecastHead(
                lookback_len=self.lookback_len,
                pred_len=self.pred_len,
                cond_dim=self.policy_cond_dim,
                hidden_dim=self.aux_forecast_hidden_dim,
                coarse_factor=self.coarse_forecast_factor,
                dropout=float(config["diffusion"].get("dropout", 0.0)),
            )
        else:
            self.coarse_forecast_head = None
        if self.use_uncertainty_head:
            self.uncertainty_head = UncertaintyHead(
                lookback_len=self.lookback_len,
                pred_len=self.pred_len,
                cond_dim=self.policy_cond_dim,
                hidden_dim=self.aux_forecast_hidden_dim,
                dropout=float(config["diffusion"].get("dropout", 0.0)),
            )
        else:
            self.uncertainty_head = None

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["decomp"] = self.decomp
        config_diff["lookback_len"] = self.lookback_len
        config_diff["pred_len"] = self.pred_len
        config_diff["with_timestep"] = True if self.timestep_emb_cat else False
        config_diff["context_dim"] = self.context_dim
        denoiser_with_text = bool(self.with_texts) and self.text_injection != "off"
        if self.text_injection == "legacy":
            denoiser_with_text = bool(self.with_texts)
        config_diff["with_texts"] = denoiser_with_text
        config_diff["text_mode"] = "legacy" if self.text_injection in ("legacy", "off") else self.text_injection
        config_diff["n_text_sources"] = self.n_text_sources
        config_diff["tower_dropout"] = config["diffusion"].get("tower_dropout", None)
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

    def _segment_slices(self, horizons=None):
        horizons = list(self.multi_res_horizons if horizons is None else horizons)
        if not horizons and self.pred_len > 0:
            horizons = [int(self.pred_len)]
        segments = []
        prev_h = 0
        for horizon in horizons:
            h = int(horizon)
            if h <= prev_h:
                continue
            start = int(self.lookback_len + prev_h)
            end = int(self.lookback_len + h)
            segments.append((h, start, end))
            prev_h = h
        return segments

    def _segment_running_mean(self, sum_buffer, count_buffer):
        count = count_buffer.detach().clamp(min=1.0)
        mean = sum_buffer.detach() / count
        return torch.where(count_buffer.detach() > 0, mean, torch.zeros_like(mean)).to(self.device)

    def get_multi_res_debug_state(self):
        active_horizons = self._get_active_multi_res_horizons()
        indices = [self.multi_res_horizon_to_index[h] for h in active_horizons if h in self.multi_res_horizon_to_index]
        difficulty = []
        if indices:
            difficulty = self.multi_res_difficulty_ema[indices].detach().cpu().tolist()
        q_b_count = float(self.pattern_q_b_count.detach().cpu().item())
        q_b_mean = float(self.pattern_q_b_sum.detach().cpu().item() / max(q_b_count, 1.0))
        w_count = float(self.diag_w_auto_count.detach().cpu().item())
        w_mean = float(self.diag_w_auto_sum.detach().cpu().item() / max(w_count, 1.0))
        w_var = float(self.diag_w_auto_sq_sum.detach().cpu().item() / max(w_count, 1.0) - w_mean ** 2)
        alpha_count = float(self.diag_alpha_b_count.detach().cpu().item())
        alpha_mean = float(self.diag_alpha_b_sum.detach().cpu().item() / max(alpha_count, 1.0))
        alpha_var = float(self.diag_alpha_b_sq_sum.detach().cpu().item() / max(alpha_count, 1.0) - alpha_mean ** 2)
        text_count = float(self.diag_text_quality_count.detach().cpu().item())
        text_gain_count = float(self.text_gain_count.detach().cpu().item())
        text_gain_batch_count = float(self.diag_text_gain_batch_count.detach().cpu().item())
        agree_count = float(self.diag_router_agreement_count.detach().cpu().item())
        baseline_count = float(self.diag_pattern_baseline_mse_count.detach().cpu().item())
        loss_count = float(self.diag_loss_count.detach().cpu().item())
        pattern_budget_count = float(self.diag_pattern_budget_count.detach().cpu().item())
        mr_budget_count = float(self.diag_mr_budget_count.detach().cpu().item())
        main_loss_mean = float(self.diag_main_loss_sum.detach().cpu().item() / max(loss_count, 1.0))
        pattern_aux_mean = float(self.diag_pattern_aux_sum.detach().cpu().item() / max(loss_count, 1.0))
        multi_res_aux_mean = float(self.diag_multi_res_aux_sum.detach().cpu().item() / max(loss_count, 1.0))
        coarse_aux_mean = float(self.diag_coarse_aux_sum.detach().cpu().item() / max(loss_count, 1.0))
        uncertainty_aux_mean = float(self.diag_uncertainty_aux_sum.detach().cpu().item() / max(loss_count, 1.0))
        segment_q_b = self._segment_running_mean(self.pattern_segment_q_b_sum, self.pattern_segment_q_b_count).detach().cpu().tolist()
        segment_alpha_count = self.diag_segment_alpha_count.detach().cpu().clamp(min=1.0)
        segment_alpha_mean_tensor = self.diag_segment_alpha_sum.detach().cpu() / segment_alpha_count
        segment_alpha_var = self.diag_segment_alpha_sq_sum.detach().cpu() / segment_alpha_count - segment_alpha_mean_tensor ** 2
        segment_alpha_mean = torch.where(self.diag_segment_alpha_count.detach().cpu() > 0, segment_alpha_mean_tensor, torch.zeros_like(segment_alpha_mean_tensor))
        segment_alpha_std = torch.where(self.diag_segment_alpha_count.detach().cpu() > 0, segment_alpha_var.clamp(min=0.0).sqrt(), torch.zeros_like(segment_alpha_var))
        segment_mse_count = self.diag_segment_baseline_mse_count.detach().cpu().clamp(min=1.0)
        segment_mse_mean = torch.where(
            self.diag_segment_baseline_mse_count.detach().cpu() > 0,
            self.diag_segment_baseline_mse_sum.detach().cpu() / segment_mse_count,
            torch.zeros_like(segment_mse_count),
        )
        return {
            "guide_mode": self.guide_mode,
            "final_method": self.final_method,
            "final_horizons": list(self.multi_res_horizons),
            "active_horizons": list(active_horizons),
            "huber_deltas": list(self.multi_res_huber_deltas),
            "horizon_reliabilities": list(self.multi_res_horizon_reliabilities),
            "segment_reliabilities": list(self.multi_res_segment_reliabilities),
            "difficulty_ema": difficulty,
            "horizon_groups": [self._get_horizon_group(horizon) for horizon in active_horizons],
            "segment_loss": self.multi_res_segment_loss,
            "difficulty_inverse": self.multi_res_difficulty_inverse,
            "pattern_q_b_running_mean": q_b_mean,
            "pattern_q_b_running_count": q_b_count,
            "segment_q_b_running_mean": segment_q_b,
            "w_auto_mean": w_mean,
            "w_auto_std": float(max(w_var, 0.0) ** 0.5),
            "text_quality_mean": float(self.diag_text_quality_sum.detach().cpu().item() / max(text_count, 1.0)),
            "text_gain_running_mean": float(self.text_gain_sum.detach().cpu().item() / max(text_gain_count, 1.0)),
            "text_gain_running_count": text_gain_count,
            "text_gain_batch_mean": float(self.diag_text_gain_batch_sum.detach().cpu().item() / max(text_gain_batch_count, 1.0)),
            "router_agreement_mean": float(self.diag_router_agreement_sum.detach().cpu().item() / max(agree_count, 1.0)),
            "alpha_b_mean": alpha_mean,
            "alpha_b_std": float(max(alpha_var, 0.0) ** 0.5),
            "segment_alpha_b_mean": segment_alpha_mean.tolist(),
            "segment_alpha_b_std": segment_alpha_std.tolist(),
            "pattern_baseline_mse": float(self.diag_pattern_baseline_mse_sum.detach().cpu().item() / max(baseline_count, 1.0)),
            "segment_pattern_baseline_mse": segment_mse_mean.tolist(),
            "pattern_budget": float(self.diag_pattern_budget_sum.detach().cpu().item() / max(pattern_budget_count, 1.0)),
            "mr_budget": float(self.diag_mr_budget_sum.detach().cpu().item() / max(mr_budget_count, 1.0)),
            "loss_means": {
                "main_loss": main_loss_mean,
                "pattern_aux": pattern_aux_mean,
                "multi_res_aux": multi_res_aux_mean,
                "coarse_aux": coarse_aux_mean,
                "uncertainty_aux": uncertainty_aux_mean,
            },
            "loss_ratios": {
                "pattern_aux_to_main": float(pattern_aux_mean / max(main_loss_mean, 1e-8)),
                "multi_res_aux_to_main": float(multi_res_aux_mean / max(main_loss_mean, 1e-8)),
                "coarse_aux_to_main": float(coarse_aux_mean / max(main_loss_mean, 1e-8)),
                "uncertainty_aux_to_main": float(uncertainty_aux_mean / max(main_loss_mean, 1e-8)),
            },
        }

    def _add_diag_scalar(self, sum_name, count_name, value, count=1.0, sq_sum_name=None):
        with torch.no_grad():
            value = value.detach().float()
            if value.numel() > 1:
                mean_value = value.mean()
                count_value = float(value.numel()) if count is None else float(count)
                sq_value = (value ** 2).mean()
            else:
                mean_value = value.reshape(()).float()
                count_value = 1.0 if count is None else float(count)
                sq_value = mean_value ** 2
            getattr(self, sum_name).add_(mean_value * count_value)
            getattr(self, count_name).add_(count_value)
            if sq_sum_name is not None:
                getattr(self, sq_sum_name).add_(sq_value * count_value)

    def _pattern_q_b_running_mean(self):
        if float(self.pattern_q_b_count.detach().item()) <= 0:
            return torch.zeros((), device=self.device)
        return (self.pattern_q_b_sum / self.pattern_q_b_count.clamp(min=1.0)).to(self.device)

    def _text_gain_running_mean(self, device=None, dtype=None):
        device = self.device if device is None else device
        if float(self.text_gain_count.detach().item()) <= 0:
            return torch.ones((), device=device, dtype=dtype or torch.float32)
        value = (self.text_gain_sum / self.text_gain_count.clamp(min=1.0)).to(device)
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value.clamp(min=0.0, max=1.0)

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
        if hist.shape[1] > 1:
            target_idx = int(min(max(self.target_feature_index, 0), hist.shape[1] - 1))
            hist = hist[:, target_idx:target_idx + 1, :]
            hist_mask = hist_mask[:, target_idx:target_idx + 1, :]
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

    def _compute_pattern_reliability(self, stats, text_pattern=None):
        if not self.pattern_reliability:
            return torch.ones((stats.shape[0],), device=stats.device, dtype=stats.dtype)
        trend_r2 = stats[:, 1]
        season_strength = stats[:, 3]
        state_score = stats[:, 4]
        stability = 1.0 - stats[:, 7].clamp(min=0.0, max=1.0)
        if self.final_method:
            return (
                torch.max(torch.stack([trend_r2, season_strength, state_score], dim=1), dim=1).values
                * torch.sqrt(stability.clamp(min=0.0, max=1.0))
            ).clamp(min=0.0, max=1.0)
        text_strength = text_pattern.max(dim=1).values if text_pattern is not None and text_pattern.numel() > 0 else torch.zeros_like(trend_r2)
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

    def _compute_text_quality(self, text_evidence_vec, text_mask, batch_size, device, dtype):
        if text_mask is None:
            availability = torch.ones((batch_size,), device=device, dtype=dtype)
        else:
            availability = text_mask.float()
            if availability.dim() > 1:
                availability = availability.mean(dim=1)
            availability = availability.reshape(-1).to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
            if availability.numel() == 1:
                availability = availability.repeat(batch_size)
        if text_evidence_vec is None:
            quality = availability
            temporal_support = availability
            source_agreement = availability
        else:
            evidence = text_evidence_vec.float()
            if evidence.dim() == 1:
                evidence = evidence.reshape(batch_size, -1)
            if evidence.shape[1] < 5:
                evidence = F.pad(evidence, (0, 5 - evidence.shape[1]))
            evidence = evidence[:, :5].to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
            quality = evidence[:, 0]
            temporal_support = torch.max(evidence[:, 1], evidence[:, 2])
            source_agreement = evidence[:, 4]
        q_t = availability * torch.pow(
            (quality * temporal_support * source_agreement).clamp(min=1e-8),
            1.0 / 3.0,
        )
        return q_t.clamp(min=0.0, max=1.0)

    def _masked_pair_corr(self, x, y, mask):
        if mask.sum() <= 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        x_sel = x[mask > 0]
        y_sel = y[mask > 0]
        x_center = x_sel - x_sel.mean()
        y_center = y_sel - y_sel.mean()
        denom = torch.sqrt((x_center.pow(2).sum() * y_center.pow(2).sum()).clamp(min=1e-6))
        return ((x_center * y_center).sum() / denom).abs().clamp(min=0.0, max=1.0)

    def _compute_covariate_utility(self, observed_data, cond_mask):
        if observed_data is None or cond_mask is None or observed_data.shape[1] <= 1:
            return torch.zeros((observed_data.shape[0] if observed_data is not None else 1,), device=self.device)
        hist = observed_data[:, :, : self.lookback_len]
        hist_mask = cond_mask[:, :, : self.lookback_len].float()
        target_idx = int(min(max(self.target_feature_index, 0), hist.shape[1] - 1))
        batch_scores = []
        for b in range(hist.shape[0]):
            target = hist[b, target_idx]
            target_mask = hist_mask[b, target_idx]
            corr_scores = []
            avail_scores = []
            for k in range(hist.shape[1]):
                if k == target_idx:
                    continue
                aux = hist[b, k]
                aux_mask = hist_mask[b, k]
                overlap = target_mask * aux_mask
                corr_scores.append(self._masked_pair_corr(target, aux, overlap))
                avail_scores.append(overlap.mean())
            if corr_scores:
                corr_tensor = torch.stack(corr_scores)
                avail_tensor = torch.stack(avail_scores)
                batch_scores.append((corr_tensor.max() * torch.sqrt(avail_tensor.mean().clamp(min=0.0, max=1.0))).clamp(0.0, 1.0))
            else:
                batch_scores.append(torch.zeros((), device=hist.device, dtype=hist.dtype))
        return torch.stack(batch_scores)

    def _compute_coarse_utility(self, observed_data, cond_mask):
        if observed_data is None or cond_mask is None:
            return torch.zeros((1,), device=self.device)
        hist = observed_data[:, self.target_feature_index:self.target_feature_index + 1, : self.lookback_len]
        hist_mask = cond_mask[:, self.target_feature_index:self.target_feature_index + 1, : self.lookback_len].float()
        time_mask = (hist_mask.sum(dim=1) > 0).float()
        series = (hist * hist_mask).sum(dim=1) / hist_mask.sum(dim=1).clamp(min=1.0)
        last = series[:, -1]
        first = series[:, 0]
        slope = (last - first).abs()
        scale = (series.std(dim=1, unbiased=False) + 1e-6)
        trend_strength = (slope / scale).clamp(min=0.0, max=2.0) / 2.0
        season_scores = []
        for lag in (2, 3, 4, 6, 7, 12, 24, 52):
            season_scores.append(self._safe_lag_corr(series, time_mask, lag).clamp(min=0.0, max=1.0))
        seasonality = torch.stack(season_scores, dim=1).max(dim=1).values if season_scores else torch.zeros_like(trend_strength)
        return torch.maximum(seasonality, trend_strength).clamp(0.0, 1.0)

    def _compute_uncertainty_utility(self, observed_data, cond_mask):
        if observed_data is None or cond_mask is None:
            return torch.zeros((1,), device=self.device)
        hist = observed_data[:, self.target_feature_index:self.target_feature_index + 1, : self.lookback_len]
        hist_mask = cond_mask[:, self.target_feature_index:self.target_feature_index + 1, : self.lookback_len].float()
        time_mask = (hist_mask.sum(dim=1) > 0).float()
        series = (hist * hist_mask).sum(dim=1) / hist_mask.sum(dim=1).clamp(min=1.0)
        diffs = series[:, 1:] - series[:, :-1]
        diff_mask = time_mask[:, 1:] * time_mask[:, :-1]
        diff_std = torch.sqrt((((diffs - (diffs * diff_mask).sum(dim=1, keepdim=True) / diff_mask.sum(dim=1, keepdim=True).clamp(min=1.0)) * diff_mask) ** 2).sum(dim=1) / diff_mask.sum(dim=1).clamp(min=1.0) + 1e-6)
        level_std = series.std(dim=1, unbiased=False).clamp(min=1e-6)
        return (diff_std / (level_std + diff_std + 1e-6)).clamp(0.0, 1.0)

    def _compute_source_utilities(self, observed_data, cond_mask, trend_prior=None, text_mask=None, retrieval_mask=None, reasoning_mask=None, text_evidence_vec=None):
        batch_size = observed_data.shape[0]
        dtype = observed_data.dtype
        device = observed_data.device
        text_utility = self._compute_text_quality(text_evidence_vec, text_mask, batch_size, device, dtype)
        if retrieval_mask is not None:
            text_utility = text_utility * retrieval_mask.to(device=device, dtype=dtype).reshape(-1)
        if reasoning_mask is not None:
            text_utility = text_utility * (0.5 + 0.5 * reasoning_mask.to(device=device, dtype=dtype).reshape(-1))
        return {
            "text": text_utility.clamp(0.0, 1.0),
            "covariate": self._compute_covariate_utility(observed_data, cond_mask).to(device=device, dtype=dtype).clamp(0.0, 1.0),
            "coarse": self._compute_coarse_utility(observed_data, cond_mask).to(device=device, dtype=dtype).clamp(0.0, 1.0),
            "uncertainty": self._compute_uncertainty_utility(observed_data, cond_mask).to(device=device, dtype=dtype).clamp(0.0, 1.0),
        }

    def _compute_text_gain(self, observed_data, cond_mask, pi_numeric, pi_text):
        if not self.training:
            return self._text_gain_running_mean(pi_numeric.device, pi_numeric.dtype)
        if observed_data is None or cond_mask is None:
            return self._text_gain_running_mean(pi_numeric.device, pi_numeric.dtype)
        hist_len = min(max(int(self.lookback_len), 1), observed_data.shape[-1])
        if hist_len <= 1:
            return self._text_gain_running_mean(pi_numeric.device, pi_numeric.dtype)
        if self.multi_res_horizons:
            cal_len = min(int(self.multi_res_horizons[0]), hist_len - 1)
        else:
            cal_len = min(max(int(self.pred_len), 1), hist_len - 1)
        cal_len = max(int(cal_len), 1)
        prefix_len = hist_len - cal_len
        if prefix_len <= 0:
            return self._text_gain_running_mean(pi_numeric.device, pi_numeric.dtype)

        prefix = observed_data[:, :, :prefix_len]
        prefix_mask = cond_mask[:, :, :prefix_len].float()
        cal_target = observed_data[:, :, prefix_len:hist_len]
        cal_mask = cond_mask[:, :, prefix_len:hist_len].float()
        if cal_mask.sum() <= 0:
            return self._text_gain_running_mean(pi_numeric.device, pi_numeric.dtype)

        expert_cal = self._build_expert_futures_from_history(prefix, prefix_mask, cal_len)
        numeric_cal = (pi_numeric.view(pi_numeric.shape[0], self.pattern_num_experts, 1, 1) * expert_cal).sum(dim=1)
        text_cal = (pi_text.view(pi_text.shape[0], self.pattern_num_experts, 1, 1) * expert_cal).sum(dim=1)
        denom = cal_mask.sum().clamp(min=1.0)
        loss_num = (((numeric_cal - cal_target) * cal_mask) ** 2).sum() / denom
        loss_text = (((text_cal - cal_target) * cal_mask) ** 2).sum() / denom
        gain = ((loss_num - loss_text) / loss_num.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
        if self.training:
            self._add_diag_scalar("text_gain_sum", "text_gain_count", gain)
        self._add_diag_scalar("diag_text_gain_batch_sum", "diag_text_gain_batch_count", gain)
        return gain.detach().to(device=pi_numeric.device, dtype=pi_numeric.dtype)

    def _compute_auto_guidance(self, numeric_logits, text_logits, text_evidence_vec, text_mask, batch_size, observed_data=None, cond_mask=None):
        pi_num = F.softmax(numeric_logits, dim=-1).clamp(min=1e-6)
        pi_text = F.softmax(text_logits, dim=-1).clamp(min=1e-6)
        kl_nt = (pi_num * (pi_num.log() - pi_text.log())).sum(dim=1)
        kl_tn = (pi_text * (pi_text.log() - pi_num.log())).sum(dim=1)
        router_agreement = torch.exp(-0.5 * (kl_nt + kl_tn)).clamp(min=0.0, max=1.0)
        text_quality = self._compute_text_quality(
            text_evidence_vec,
            text_mask,
            batch_size,
            numeric_logits.device,
            numeric_logits.dtype,
        )
        text_gain = self._compute_text_gain(observed_data, cond_mask, pi_num, pi_text)
        w_auto = (text_quality * router_agreement * text_gain).clamp(min=0.0, max=1.0)
        self._add_diag_scalar("diag_w_auto_sum", "diag_w_auto_count", w_auto, count=None, sq_sum_name="diag_w_auto_sq_sum")
        self._add_diag_scalar("diag_text_quality_sum", "diag_text_quality_count", text_quality, count=None)
        self._add_diag_scalar("diag_router_agreement_sum", "diag_router_agreement_count", router_agreement, count=None)
        return w_auto, text_quality, router_agreement, pi_num, pi_text, text_gain

    def _build_fpp4_expert_futures(self, hist, hist_mask, future_len):
        """Mean, naive, seasonal naive, and drift on the normalized history."""
        B, K, hist_len = hist.shape
        if future_len <= 0:
            return torch.zeros((B, 4, K, 0), device=hist.device, dtype=hist.dtype)
        mean = self._masked_mean(hist, hist_mask, dim=2, keepdim=True)
        filled_hist = torch.where(hist_mask > 0, hist, mean.expand_as(hist))
        first = filled_hist[:, :, 0:1]
        last = filled_hist[:, :, -1:]
        steps = torch.arange(1, future_len + 1, device=hist.device, dtype=hist.dtype).view(1, 1, -1)
        mean_future = mean.expand(-1, -1, future_len)
        naive_future = last.expand(-1, -1, future_len)
        period = int(self.season_period)
        if hist_len > period > 0:
            season_index = hist_len - period + (torch.arange(future_len, device=hist.device) % period)
            seasonal_future = filled_hist[:, :, season_index]
        else:
            seasonal_future = naive_future
        drift_future = last + steps * (last - first) / max(hist_len - 1, 1)
        return torch.stack([mean_future, naive_future, seasonal_future, drift_future], dim=1)

    def _linear_history_router(self, hist, hist_mask, text_pooled=None, text_mask=None):
        target_idx = int(min(max(self.target_feature_index, 0), hist.shape[1] - 1))
        series = hist[:, target_idx, :]
        mask = hist_mask[:, target_idx, :]
        series = torch.where(mask > 0, series, torch.zeros_like(series))
        if series.shape[-1] < self.lookback_len:
            series = F.pad(series, (0, self.lookback_len - series.shape[-1]))
        elif series.shape[-1] > self.lookback_len:
            series = series[:, : self.lookback_len]
        logits = self.pattern_router_linear(series)
        if (
            self.pattern_router_mode == "linear_history_text"
            and self.pattern_text_head is not None
            and text_pooled is not None
        ):
            text_logits = self.pattern_text_head(text_pooled.float())
            if text_mask is not None:
                availability = text_mask.float().reshape(-1, 1).to(device=text_logits.device, dtype=text_logits.dtype)
                text_logits = text_logits * availability
            logits = logits + text_logits
        temperature = max(float(self.pattern_router_temperature), 1e-3)
        return F.softmax(logits / temperature, dim=-1)

    def _empty_pattern_result(self, observed_data):
        batch = observed_data.shape[0]
        device = observed_data.device
        dtype = observed_data.dtype
        zeros_b = torch.zeros((batch,), device=device, dtype=dtype)
        return {
            "stats": torch.zeros((batch, self.pattern_stats_dim), device=device, dtype=dtype),
            "text_pattern": torch.zeros((batch, self.pattern_num_experts), device=device, dtype=dtype),
            "pi_numeric": torch.zeros((batch, self.pattern_num_experts), device=device, dtype=dtype),
            "pi_text": torch.zeros((batch, self.pattern_num_experts), device=device, dtype=dtype),
            "pi": torch.zeros((batch, self.pattern_num_experts), device=device, dtype=dtype),
            "pseudo": torch.full((batch, self.pattern_num_experts), 1.0 / max(self.pattern_num_experts, 1), device=device, dtype=dtype),
            "baseline": torch.zeros_like(observed_data),
            "expert_futures": torch.zeros((batch, self.pattern_num_experts, observed_data.shape[1], 0), device=device, dtype=dtype),
            "baseline_reliability": torch.ones_like(zeros_b),
            "text_quality": zeros_b,
            "router_agreement": torch.ones_like(zeros_b),
            "text_gain": torch.ones((), device=device, dtype=dtype),
            "guidance_scale": torch.ones_like(zeros_b),
        }

    def _compute_simple_baseline_outputs(self, observed_data, cond_mask, text_pooled=None, text_mask=None):
        result = self._empty_pattern_result(observed_data)
        B, K, L = observed_data.shape
        future_len = min(max(int(self.pred_len), 0), max(L - self.lookback_len, 0))
        hist_len = min(max(int(self.lookback_len), 1), L)
        hist = observed_data[:, :, :hist_len]
        hist_mask = cond_mask[:, :, :hist_len].float()
        baseline = torch.zeros_like(observed_data)
        if future_len <= 0:
            result["baseline"] = baseline
            return result
        if self.baseline_form == "dlinear" and self.dlinear_baseline is not None:
            future = self.dlinear_baseline(hist)[:, :, :future_len]
            baseline[:, :, self.lookback_len:self.lookback_len + future_len] = future
            result["baseline"] = baseline.detach() if self.pattern_baseline_detach else baseline
            # One deterministic expert so the same Huber term trains DLinear.
            result["pi"] = torch.ones((B, 1), device=observed_data.device, dtype=observed_data.dtype)
            result["expert_futures"] = future.unsqueeze(1)
            return result
        if self.pattern_expert_set == "fpp4" or self.pattern_router_mode != "legacy":
            experts = self._build_fpp4_expert_futures(hist, hist_mask, future_len)
        else:
            experts = self._build_expert_futures_from_history(hist, hist_mask, future_len)
        if self.baseline_form == "naive":
            naive_index = 1 if experts.shape[1] == 4 else min(2, experts.shape[1] - 1)
            pi = torch.zeros((B, experts.shape[1]), device=observed_data.device, dtype=observed_data.dtype)
            pi[:, naive_index] = 1.0
        else:
            pi = self._linear_history_router(hist, hist_mask, text_pooled=text_pooled, text_mask=text_mask)
        mixed = (pi.view(B, -1, 1, 1) * experts).sum(dim=1)
        baseline[:, :, self.lookback_len:self.lookback_len + future_len] = mixed
        if self.pattern_baseline_detach:
            baseline = baseline.detach()
        result["baseline"] = baseline
        result["pi"] = pi
        result["pi_numeric"] = pi
        result["pi_text"] = pi
        result["expert_futures"] = experts
        return result

    def _build_expert_futures_from_history(self, hist, hist_mask, future_len):
        if self.pattern_expert_set == "fpp4":
            return self._build_fpp4_expert_futures(hist, hist_mask, future_len)
        B, K, hist_len = hist.shape
        if future_len <= 0:
            return torch.zeros((B, self.pattern_num_experts, K, 0), device=hist.device, dtype=hist.dtype)
        mean = self._masked_mean(hist, hist_mask, dim=2, keepdim=True)
        filled_hist = torch.where(hist_mask > 0, hist, mean.expand_as(hist))
        first = filled_hist[:, :, 0:1]
        last = filled_hist[:, :, -1:]
        prev = filled_hist[:, :, -2:-1] if hist_len > 1 else last
        slope = (last - first) / max(hist_len - 1, 1)
        steps = torch.arange(1, future_len + 1, device=hist.device, dtype=hist.dtype).view(1, 1, -1)
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
            idx = hist_len - p + (torch.arange(future_len, device=hist.device) % p)
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
        return expert_futures

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
        expert_futures = self._build_expert_futures_from_history(hist, hist_mask, future_len)
        return baseline, expert_futures

    def _compute_pattern_outputs(self, observed_data, cond_mask, text_pooled=None, text_mask=None, text_evidence_vec=None, guidance_scale=1.0, controller_state=None):
        if self.pattern_router_mode != "legacy" or self.baseline_form in ("naive", "dlinear"):
            return self._compute_simple_baseline_outputs(
                observed_data,
                cond_mask,
                text_pooled=text_pooled,
                text_mask=text_mask,
            )
        B, K, L = observed_data.shape
        stats = self._compute_pattern_stats(observed_data, cond_mask)
        text_pattern = self._compute_pattern_text_evidence(text_pooled, text_mask, text_evidence_vec, B, observed_data.device)
        text_pattern = self._maybe_drop_pattern_text(text_pattern)
        temperature = max(float(self.pattern_router_temperature), 1e-3)
        numeric_router_input = torch.cat([stats, torch.zeros_like(text_pattern)], dim=1)
        text_router_input = torch.cat([stats, text_pattern], dim=1)
        numeric_logits = self.pattern_router(numeric_router_input) / temperature
        text_logits = self.pattern_router(text_router_input) / temperature
        controller_scale = None
        controller_residual_scale = None
        if controller_state is not None:
            controller_scale = controller_state.get("guidance_scale")
            controller_residual_scale = controller_state.get("residual_scale")
        if controller_scale is None:
            controller_scale = torch.ones((B,), device=observed_data.device, dtype=observed_data.dtype)
        else:
            controller_scale = controller_scale.to(device=observed_data.device, dtype=observed_data.dtype).reshape(-1)
        if self.guide_mode == "auto":
            scale, text_quality, router_agreement, pi_numeric, pi_text, text_gain = self._compute_auto_guidance(
                numeric_logits,
                text_logits,
                text_evidence_vec,
                text_mask,
                B,
                observed_data=observed_data,
                cond_mask=cond_mask,
            )
            scale = scale * controller_scale
        else:
            scale = torch.full((B,), float(guidance_scale), device=observed_data.device, dtype=observed_data.dtype) * controller_scale
            text_quality = self._compute_text_quality(text_evidence_vec, text_mask, B, observed_data.device, observed_data.dtype)
            router_agreement = torch.ones((B,), device=observed_data.device, dtype=observed_data.dtype)
            pi_numeric = F.softmax(numeric_logits, dim=-1)
            pi_text = F.softmax(text_logits, dim=-1)
            text_gain = torch.ones((), device=observed_data.device, dtype=observed_data.dtype)
        guided_logits = numeric_logits + scale.reshape(B, 1) * (text_logits - numeric_logits)
        pi = F.softmax(guided_logits, dim=-1)
        baseline, expert_futures = self._build_pattern_baselines(observed_data, cond_mask, stats)
        future_len = expert_futures.shape[-1]
        reliability = self._compute_pattern_reliability(stats, text_pattern)
        if future_len > 0:
            mixed_future = (pi.view(B, self.pattern_num_experts, 1, 1) * expert_futures).sum(dim=1)
            safe_future = expert_futures[:, 2]
            mixed_future = reliability.view(B, 1, 1) * mixed_future + (1.0 - reliability).view(B, 1, 1) * safe_future
            if controller_residual_scale is not None:
                mixed_future = mixed_future * controller_residual_scale.to(
                    device=observed_data.device,
                    dtype=observed_data.dtype,
                ).reshape(B, 1, 1)
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
            "pi_numeric": pi_numeric,
            "pi_text": pi_text,
            "pi": pi,
            "pseudo": pseudo.detach(),
            "baseline": baseline,
            "expert_futures": expert_futures,
            "baseline_reliability": reliability,
            "text_quality": text_quality,
            "router_agreement": router_agreement,
            "text_gain": text_gain,
            "guidance_scale": scale,
        }

    def _compute_weak_pattern_baseline(self, pattern, observed_data, target_mask, use_training_target):
        baseline = pattern["baseline"]
        reliability = pattern.get("baseline_reliability")
        if reliability is None:
            reliability = torch.ones((baseline.shape[0],), device=baseline.device, dtype=baseline.dtype)
        reliability = reliability.reshape(-1).clamp(min=0.0, max=1.0)
        segments = self._segment_slices()
        if not segments:
            alpha = torch.zeros((baseline.shape[0], 0), device=baseline.device, dtype=baseline.dtype)
            pattern["alpha_b"] = torch.zeros_like(reliability)
            pattern["alpha_b_segment"] = alpha
            return baseline * 0.0

        running_q = self._segment_running_mean(self.pattern_segment_q_b_sum, self.pattern_segment_q_b_count).to(
            device=baseline.device,
            dtype=baseline.dtype,
        )
        segment_reliability = self.multi_res_segment_reliability.detach().to(device=baseline.device, dtype=baseline.dtype).clamp(min=0.0, max=1.0)
        scaled = baseline.clone()
        segment_q_values = []
        segment_alpha_values = []
        segment_mse_values = []
        max_segments = int(self.pattern_segment_q_b_sum.numel())
        for segment_idx, (_, start, end) in enumerate(segments):
            if segment_idx >= max_segments:
                continue
            end = min(end, observed_data.shape[-1])
            if end <= start:
                continue
            future_baseline = baseline[:, :, start:end]
            if use_training_target:
                future_mask = target_mask[:, :, start:end].float()
                future_target = observed_data[:, :, start:end]
                if future_mask.sum() > 0:
                    denom = future_mask.sum().clamp(min=1.0)
                    baseline_mse = (((future_baseline - future_target) * future_mask) ** 2).sum() / denom
                    target_mean = (future_target * future_mask).sum() / denom
                    target_var = (((future_target - target_mean) * future_mask) ** 2).sum() / denom
                    q_b = (1.0 - baseline_mse / target_var.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
                    segment_mse_values.append(baseline_mse.detach())
                    self._add_diag_scalar("diag_pattern_baseline_mse_sum", "diag_pattern_baseline_mse_count", baseline_mse)
                    with torch.no_grad():
                        self.diag_segment_baseline_mse_sum[segment_idx].add_(baseline_mse.detach().float())
                        self.diag_segment_baseline_mse_count[segment_idx].add_(1.0)
                        if self.training:
                            self.pattern_segment_q_b_sum[segment_idx].add_(q_b.detach().float())
                            self.pattern_segment_q_b_count[segment_idx].add_(1.0)
                            self.pattern_q_b_sum.add_(q_b.detach().float())
                            self.pattern_q_b_count.add_(1.0)
                else:
                    q_b = running_q[segment_idx] if segment_idx < running_q.numel() else torch.zeros((), device=baseline.device, dtype=baseline.dtype)
            else:
                q_b = running_q[segment_idx] if segment_idx < running_q.numel() else torch.zeros((), device=baseline.device, dtype=baseline.dtype)
            rel_s = segment_reliability[segment_idx] if segment_idx < segment_reliability.numel() else torch.ones((), device=baseline.device, dtype=baseline.dtype)
            alpha_s = (q_b.detach() * reliability * rel_s).clamp(min=0.0, max=1.0)
            scaled[:, :, start:end] = future_baseline * alpha_s.reshape(-1, 1, 1)
            segment_q_values.append(q_b.detach().reshape(1))
            segment_alpha_values.append(alpha_s)
            with torch.no_grad():
                self.diag_segment_alpha_sum[segment_idx].add_(alpha_s.detach().float().mean())
                self.diag_segment_alpha_sq_sum[segment_idx].add_((alpha_s.detach().float() ** 2).mean())
                self.diag_segment_alpha_count[segment_idx].add_(1.0)

        if segment_alpha_values:
            alpha_tensor = torch.stack(segment_alpha_values, dim=1)
            alpha = alpha_tensor.mean(dim=1)
            q_tensor = torch.cat(segment_q_values, dim=0)
        else:
            alpha_tensor = torch.zeros((baseline.shape[0], 0), device=baseline.device, dtype=baseline.dtype)
            alpha = torch.zeros_like(reliability)
            q_tensor = torch.zeros((0,), device=baseline.device, dtype=baseline.dtype)
        pattern["alpha_b"] = alpha
        pattern["alpha_b_segment"] = alpha_tensor
        pattern["pattern_q_b"] = q_tensor.detach()
        if segment_mse_values:
            pattern["segment_pattern_baseline_mse"] = torch.stack(segment_mse_values).detach()
        self._add_diag_scalar("diag_alpha_b_sum", "diag_alpha_b_count", alpha_tensor if alpha_tensor.numel() > 0 else alpha, count=None, sq_sum_name="diag_alpha_b_sq_sum")
        return scaled

    def _calc_simple_expert_loss(self, pattern, observed_data, target_mask):
        aux = torch.zeros((), device=observed_data.device)
        if pattern is None or self.pattern_expert_weight <= 0:
            return aux
        experts = pattern.get("expert_futures")
        pi = pattern.get("pi")
        if experts is None or pi is None or experts.numel() == 0 or pi.numel() == 0 or pi.shape[-1] == 0:
            return aux
        future_len = experts.shape[-1]
        start = int(self.lookback_len)
        end = start + future_len
        future_mask = target_mask[:, :, start:end].unsqueeze(1)
        future_target = observed_data[:, :, start:end].unsqueeze(1)
        residual = (experts - future_target) * future_mask
        huber = self._smooth_l1(residual)
        denom = future_mask.sum(dim=(2, 3)).clamp(min=1.0)
        expert_loss = huber.sum(dim=(2, 3)) / denom
        weighted = (pi * expert_loss).sum(dim=1).mean()
        return self.pattern_expert_weight * weighted

    def _calc_pattern_aux_loss(self, pattern, observed_data, target_mask):
        if self.pattern_router_mode != "legacy" or self.baseline_form in ("naive", "dlinear"):
            return self._calc_simple_expert_loss(pattern, observed_data, target_mask)
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
        consistency = (pseudo * (pseudo.log() - pi.log())).sum(dim=1)
        consistency = (reliability * consistency).mean()
        aux = aux + consistency
        if pattern["expert_futures"].numel() > 0:
            future_len = pattern["expert_futures"].shape[-1]
            start = int(self.lookback_len)
            end = start + future_len
            future_mask = target_mask[:, :, start:end].unsqueeze(1)
            future_target = observed_data[:, :, start:end].unsqueeze(1)
            residual = (pattern["expert_futures"] - future_target) * future_mask
            huber = self._smooth_l1(residual)
            denom = future_mask.sum(dim=(2, 3)).clamp(min=1.0)
            expert_loss = huber.sum(dim=(2, 3)) / denom
            weighted_expert_loss = (pattern["pi"] * expert_loss).sum(dim=1)
            weighted_expert_loss = (reliability * weighted_expert_loss).mean()
            aux = aux + weighted_expert_loss
        if not self.final_method:
            legacy_aux = torch.zeros((), device=observed_data.device)
            if self.pattern_consistency_weight > 0:
                legacy_aux = legacy_aux + self.pattern_consistency_weight * consistency
            if self.pattern_expert_weight > 0 and pattern["expert_futures"].numel() > 0:
                legacy_aux = legacy_aux + self.pattern_expert_weight * weighted_expert_loss
            return legacy_aux
        return aux

    def _smooth_l1(self, residual):
        """Huber penalty. δ is multi_res_huber_delta (default 1, the old literal)."""
        delta = float(self.multi_res_huber_delta)
        abs_res = residual.abs()
        return torch.where(
            abs_res <= delta,
            0.5 * residual.square(),
            delta * abs_res - 0.5 * (delta ** 2),
        )

    def _per_sample_target_mse(self, prediction, target, target_mask):
        residual = (target - prediction) * target_mask
        denom = target_mask.sum(dim=(1, 2)).clamp(min=1.0)
        return (residual ** 2).sum(dim=(1, 2)) / denom

    def _current_gate_constant(self):
        if float(self.gate_label_count.detach()) <= 0:
            return self.gate_label_sum.new_tensor(0.5)
        return (self.gate_label_sum / self.gate_label_count.clamp(min=1.0)).clamp(0.0, 1.0)

    def _update_gate_label_mean(self, label, valid):
        if not self.training or label is None or valid is None or not bool(valid.any()):
            return
        with torch.no_grad():
            self.gate_label_sum.add_(label.detach()[valid].float().sum())
            self.gate_label_count.add_(valid.float().sum())

    def _text_gate_value(self, similarity, source_mask):
        batch = int(similarity.shape[0])
        if source_mask is None:
            source_mask = similarity.new_zeros((batch, self.n_text_sources))
        else:
            source_mask = source_mask.to(device=similarity.device, dtype=similarity.dtype)
            if source_mask.shape[-1] < self.n_text_sources:
                source_mask = F.pad(source_mask, (0, self.n_text_sources - source_mask.shape[-1]))
            elif source_mask.shape[-1] > self.n_text_sources:
                source_mask = source_mask[:, : self.n_text_sources]
        any_mask = (source_mask.sum(dim=-1) > 0).to(dtype=similarity.dtype)
        if self.text_gate_mode == "one":
            return any_mask
        if self.text_gate_mode == "constant":
            constant = self._current_gate_constant().to(device=similarity.device, dtype=similarity.dtype)
            return any_mask * constant
        if self.text_gate is None:
            return any_mask
        return self.text_gate(similarity.detach(), source_mask, any_mask)

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, timesteps=None, timestep_emb=None, size_emb=None, context=None, trend_prior=None, pattern_text_pooled=None, pattern_text_mask=None, pattern_text_evidence_vec=None, controller_state=None, text_pack=None
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=context, trend_prior=trend_prior, pattern_text_pooled=pattern_text_pooled, pattern_text_mask=pattern_text_mask, pattern_text_evidence_vec=pattern_text_evidence_vec, controller_state=controller_state, text_pack=text_pack
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, timesteps=None, timestep_emb=None, size_emb=None, context=None, trend_prior=None, pattern_text_pooled=None, pattern_text_mask=None, pattern_text_evidence_vec=None, controller_state=None, set_t=-1, text_pack=None
    ):

        B, K, L = observed_data.shape
        if not self.noise_esti:
            cond_count = torch.sum(cond_mask, dim=2, keepdim=True)
            safe_count = cond_count.clamp(min=1.0)
            means = torch.sum(observed_data * cond_mask, dim=2, keepdim=True) / safe_count
            var_denom = (cond_count - 1.0).clamp(min=1.0)
            stdev = torch.sqrt(torch.sum((observed_data - means) ** 2 * cond_mask, dim=2, keepdim=True) / var_denom + 1e-5)
            observed_data = (observed_data - means) / stdev

        target_mask = observed_mask - cond_mask
        pattern = None
        pattern_baseline = None
        aux_outputs = None
        diffusion_target = observed_data
        if not self.noise_esti:
            aux_outputs = self._compute_aux_head_outputs(
                observed_data,
                cond_mask,
                trend_prior=trend_prior,
                controller_state=controller_state,
                text_mask=pattern_text_mask,
                text_evidence_vec=pattern_text_evidence_vec,
            )
        if self.pattern_residual_diffusion and not self.noise_esti:
            pattern = self._compute_pattern_outputs(
                observed_data,
                cond_mask,
                text_pooled=pattern_text_pooled,
                text_mask=pattern_text_mask,
                text_evidence_vec=pattern_text_evidence_vec,
                guidance_scale=1.0,
                controller_state=controller_state,
            )
            if self.final_method and self.pattern_router_mode == "legacy" and self.baseline_form not in ("naive", "dlinear"):
                pattern_baseline = self._compute_weak_pattern_baseline(
                    pattern,
                    observed_data,
                    target_mask,
                    use_training_target=(is_train == 1),
                )
            else:
                pattern_baseline = pattern["baseline"]
            aux_future_baseline = self._build_aux_future_baseline(
                aux_outputs,
                controller_state,
                observed_data.device,
                observed_data.dtype,
                observed_data.shape[-1],
            )
            if aux_future_baseline is not None:
                pattern_baseline = aux_future_baseline if pattern_baseline is None else pattern_baseline + aux_future_baseline
            diffusion_target = observed_data - pattern_baseline
        elif not self.noise_esti:
            aux_future_baseline = self._build_aux_future_baseline(
                aux_outputs,
                controller_state,
                observed_data.device,
                observed_data.dtype,
                observed_data.shape[-1],
            )
            if aux_future_baseline is not None:
                pattern_baseline = aux_future_baseline
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

        want_similarity = (
            self.text_injection == "gated"
            and text_pack is not None
            and is_train == 1
            and self.text_gate_mode in ("learned", "constant")
        )
        predicted, text_similarity = self._run_diffusion_model(
            total_input,
            side_info,
            t,
            cfg_mask,
            timestep_emb=timestep_emb,
            size_emb=size_emb,
            context=context,
            text_pack=text_pack,
            return_text_stats=want_similarity,
        )

        validity_loss = torch.zeros((), device=observed_data.device)
        if want_similarity and text_similarity is not None and cfg_mask is not None:
            with torch.no_grad():
                reference_mask = 1.0 - cfg_mask.reshape(-1)
                reference, _ = self._run_diffusion_model(
                    total_input,
                    side_info,
                    t,
                    reference_mask,
                    timestep_emb=timestep_emb,
                    size_emb=size_emb,
                    context=context,
                    text_pack=text_pack,
                    return_text_stats=False,
                )
            injected = cfg_mask.reshape(-1).float()
            loss_main_b = self._per_sample_target_mse(predicted.detach(), diffusion_target, target_mask)
            loss_ref_b = self._per_sample_target_mse(reference, diffusion_target, target_mask)
            loss_cond = torch.where(injected > 0.5, loss_main_b, loss_ref_b)
            loss_null = torch.where(injected > 0.5, loss_ref_b, loss_main_b)
            label = benefit_label(loss_null, loss_cond, eps=self.benefit_eps)
            source_mask = text_pack.get("source_mask")
            if source_mask is None:
                any_valid = torch.ones((observed_data.shape[0],), device=observed_data.device, dtype=torch.bool)
            else:
                any_valid = source_mask.reshape(observed_data.shape[0], -1).sum(dim=-1) > 0
            self._update_gate_label_mean(label, any_valid)
            if self.text_gate_mode == "learned" and self.text_validity_loss and self.text_gate is not None and bool(any_valid.any()):
                gate = self._text_gate_value(text_similarity.detach(), source_mask)
                validity_loss = F.binary_cross_entropy(
                    gate[any_valid].clamp(1e-6, 1.0 - 1e-6),
                    label[any_valid].clamp(0.0, 1.0),
                )

        if self.timestep_outer_mix != "none" and self.timestep_branch and timesteps is not None:
            predicted_from_timestep = self.timestep_pred(timesteps)
            predicted = 0.9 * predicted + 0.1 * predicted_from_timestep

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
            pattern_aux = self._calc_pattern_aux_loss(pattern, observed_data, target_mask)
            if controller_state is not None and controller_state.get("pattern_aux_scale") is not None:
                pattern_aux = pattern_aux * controller_state["pattern_aux_scale"].to(device=observed_data.device, dtype=observed_data.dtype).mean()
            if self.final_method:
                pattern_budget = self._pattern_aux_budget(pattern, observed_data.device, observed_data.dtype)
                pattern_aux = pattern_budget * main_loss.detach() * pattern_aux / pattern_aux.detach().clamp(min=1e-6)
            auxiliary_loss = auxiliary_loss + pattern_aux
        else:
            pattern_aux = torch.zeros((), device=observed_data.device)
            pattern_budget = torch.zeros((), device=observed_data.device, dtype=observed_data.dtype)
        if (not self.noise_esti) and self.multi_res_loss_weight > 0 and len(self.multi_res_horizons) > 0:
            aux_loss = self._calc_multi_res_loss(observed_data, predicted_series, target_mask, t=t, trend_prior=trend_prior)
            if controller_state is not None and controller_state.get("multi_res_aux_scale") is not None:
                aux_loss = aux_loss * controller_state["multi_res_aux_scale"].to(device=observed_data.device, dtype=observed_data.dtype).mean()
            if self.final_method:
                mr_budget = self._multi_res_aux_budget(observed_data.device, observed_data.dtype)
                multi_res_aux = mr_budget * main_loss.detach() * aux_loss / aux_loss.detach().clamp(min=1e-6)
                auxiliary_loss = auxiliary_loss + multi_res_aux
            else:
                multi_res_aux = self.multi_res_loss_weight * aux_loss
                auxiliary_loss = auxiliary_loss + multi_res_aux
        else:
            multi_res_aux = torch.zeros((), device=observed_data.device)
            mr_budget = torch.zeros((), device=observed_data.device, dtype=observed_data.dtype)
        coarse_aux_loss = self._calc_coarse_aux_loss(aux_outputs, observed_data, target_mask, controller_state=controller_state)
        uncertainty_aux_loss = self._calc_uncertainty_aux_loss(aux_outputs, observed_data, target_mask, controller_state=controller_state)
        coarse_aux = self.coarse_forecast_weight * coarse_aux_loss if self.coarse_forecast_weight > 0 else torch.zeros((), device=observed_data.device)
        uncertainty_aux = self.uncertainty_forecast_weight * uncertainty_aux_loss if self.uncertainty_forecast_weight > 0 else torch.zeros((), device=observed_data.device)
        auxiliary_loss = auxiliary_loss + coarse_aux + uncertainty_aux
        aux_diag_scale = torch.ones((), device=observed_data.device)
        if self.auxiliary_loss_max_ratio > 0:
            aux_cap = max(self.auxiliary_loss_max_ratio, 0.0) * main_loss.detach()
            capped_auxiliary_loss = torch.minimum(auxiliary_loss, aux_cap)
            aux_diag_scale = capped_auxiliary_loss.detach() / auxiliary_loss.detach().clamp(min=1e-6)
            auxiliary_loss = capped_auxiliary_loss
        with torch.no_grad():
            self.diag_main_loss_sum.add_(main_loss.detach().float())
            self.diag_pattern_aux_sum.add_((pattern_aux.detach() * aux_diag_scale).float())
            self.diag_multi_res_aux_sum.add_((multi_res_aux.detach() * aux_diag_scale).float())
            self.diag_coarse_aux_sum.add_((coarse_aux.detach() * aux_diag_scale).float())
            self.diag_uncertainty_aux_sum.add_((uncertainty_aux.detach() * aux_diag_scale).float())
            self.diag_loss_count.add_(1.0)
        if is_train != 1 and self.valid_loss_main_only:
            return main_loss
        return main_loss + auxiliary_loss + validity_loss

    def _unwrap_diffmodel_output(self, output):
        if isinstance(output, tuple):
            return output[0]
        return output

    def _invoke_denoiser(self, model, total_input, side_info, t, cfg_mask, timestep_emb, size_emb, context, text_pack, return_text_stats):
        text_hidden = None if text_pack is None else text_pack.get("hidden")
        text_token_mask = None if text_pack is None else text_pack.get("token_mask")
        text_source_ids = None if text_pack is None else text_pack.get("source_ids")
        output = model(
            total_input,
            side_info,
            t,
            cfg_mask,
            timestep_emb,
            size_emb,
            context,
            text_hidden=text_hidden,
            text_token_mask=text_token_mask,
            text_source_ids=text_source_ids,
            return_text_stats=return_text_stats,
        )
        if return_text_stats:
            if self.save_attn:
                predicted, _, similarity = output
            else:
                predicted, similarity = output
            return predicted, similarity
        return self._unwrap_diffmodel_output(output), None

    def _run_diffusion_model(self, total_input, side_info, t, cfg_mask, timestep_emb=None, size_emb=None, context=None, text_pack=None, return_text_stats=False):
        if self.decomp:
            predicted_seasonal, similarity_seasonal = self._invoke_denoiser(
                self.diffmodel_sesonal, total_input[0], side_info, t, cfg_mask, timestep_emb, size_emb, context, text_pack, return_text_stats
            )
            predicted_trend, similarity_trend = self._invoke_denoiser(
                self.diffmodel_trend, total_input[1], side_info, t, cfg_mask, timestep_emb, size_emb, context, text_pack, return_text_stats
            )
            similarity = None
            if return_text_stats and similarity_seasonal is not None and similarity_trend is not None:
                similarity = 0.5 * (similarity_seasonal + similarity_trend)
            return predicted_seasonal + predicted_trend, similarity
        if self.save_attn and not return_text_stats:
            predicted, _ = self.diffmodel(total_input, side_info, t, cfg_mask, timestep_emb, size_emb, context)
            return predicted, None
        return self._invoke_denoiser(
            self.diffmodel, total_input, side_info, t, cfg_mask, timestep_emb, size_emb, context, text_pack, return_text_stats
        )

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

    def _active_segment_reliabilities(self, horizons):
        active_indices = [
            self.multi_res_horizon_to_index[horizon]
            for horizon in horizons
            if horizon in self.multi_res_horizon_to_index
        ]
        if len(active_indices) != len(horizons) or self.multi_res_segment_reliability.numel() < len(self.multi_res_horizons):
            return torch.ones((len(horizons),), device=self.device)
        return self.multi_res_segment_reliability[active_indices].detach().clamp(min=0.0, max=1.0).to(self.device)

    def _active_segment_reliability_mean(self):
        horizons = self._get_active_multi_res_horizons()
        if not horizons:
            return torch.ones((), device=self.device)
        return self._active_segment_reliabilities(horizons).mean().clamp(min=0.0, max=1.0)

    def _pattern_aux_budget(self, pattern, device, dtype):
        alpha_segments = None if pattern is None else pattern.get("alpha_b_segment")
        if alpha_segments is not None and torch.is_tensor(alpha_segments) and alpha_segments.numel() > 0:
            budget = alpha_segments.detach().mean()
        elif pattern is not None and torch.is_tensor(pattern.get("alpha_b")):
            budget = pattern["alpha_b"].detach().mean()
        else:
            budget = torch.zeros((), device=device, dtype=dtype)
        budget = budget.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
        self._add_diag_scalar("diag_pattern_budget_sum", "diag_pattern_budget_count", budget)
        return budget

    def _multi_res_aux_budget(self, device, dtype):
        horizons = self._get_active_multi_res_horizons()
        if not horizons:
            return torch.zeros((), device=device, dtype=dtype)
        active_indices = [
            self.multi_res_horizon_to_index[horizon]
            for horizon in horizons
            if horizon in self.multi_res_horizon_to_index
        ]
        if len(active_indices) != len(horizons):
            return torch.ones((), device=device, dtype=dtype)
        reliability = self._active_segment_reliabilities(horizons).to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
        segment_q = self._segment_running_mean(self.pattern_segment_q_b_sum, self.pattern_segment_q_b_count)
        segment_q = segment_q[active_indices].to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
        reliability_mean = reliability.mean()
        segment_gain_mean = segment_q.mean() if segment_q.numel() > 0 else torch.zeros((), device=device, dtype=dtype)
        effective_segment_ratio = reliability_mean
        budget = (reliability_mean * segment_gain_mean * effective_segment_ratio).clamp(min=0.0, max=1.0)
        self._add_diag_scalar("diag_mr_budget_sum", "diag_mr_budget_count", budget)
        return budget

    def _trend_segment_modulation(self, horizons, batch_size, trend_prior=None):
        if trend_prior is None:
            return torch.ones((batch_size, len(horizons)), device=self.device)
        if trend_prior.dim() == 3:
            strength = trend_prior[:, :, 1].mean(dim=1)
            volatility = trend_prior[:, :, 2].mean(dim=1)
        else:
            strength = trend_prior[:, 1]
            volatility = trend_prior[:, 2]
        strength = strength.reshape(-1).float().to(self.device)
        volatility = volatility.reshape(-1).float().to(self.device)
        if strength.numel() == 1:
            strength = strength.repeat(batch_size)
        if volatility.numel() == 1:
            volatility = volatility.repeat(batch_size)
        z_strength = (strength - strength.mean()) / strength.std(unbiased=False).clamp(min=1e-6)
        z_volatility = (volatility - volatility.mean()) / volatility.std(unbiased=False).clamp(min=1e-6)
        horizon_pos = torch.tensor(horizons, device=self.device, dtype=torch.float32) / max(float(self.pred_len), 1.0)
        raw = F.softplus(1.0 + horizon_pos.unsqueeze(0) * (z_strength - z_volatility).unsqueeze(1)).clamp(min=1e-6)
        if self.final_method:
            reliability = self._active_segment_reliabilities(horizons).unsqueeze(0).to(raw.device, raw.dtype)
            raw = 1.0 + reliability * (raw - 1.0)
        return raw.clamp(min=1e-6)

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

        if self.final_method:
            segment_reliability = self._active_segment_reliabilities(horizons)
            trend_modulation = self._trend_segment_modulation(horizons, batch_size, trend_prior=trend_prior)
            difficulty = self.multi_res_difficulty_ema[active_indices].detach().clamp(min=1e-6)
            if self.multi_res_difficulty_inverse:
                if self.multi_res_difficulty_gamma_configured:
                    gamma = torch.tensor(max(float(self.multi_res_difficulty_gamma), 0.0), device=self.device)
                else:
                    gamma = difficulty.std(unbiased=False) / (difficulty.mean() + difficulty.std(unbiased=False) + 1e-6)
                difficulty_term = (difficulty.mean().clamp(min=1e-6) / difficulty).pow(gamma.clamp(min=0.0))
            else:
                difficulty_term = torch.ones_like(difficulty)
            difficulty_term = segment_reliability + (1.0 - segment_reliability) * difficulty_term
            weights = segment_reliability.unsqueeze(0) * trend_modulation * difficulty_term.unsqueeze(0)
            return weights.clamp(min=1e-6)

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
            if self.mask_noisy_target:
                noisy_target = noisy_target * (1.0 - cond_mask).unsqueeze(1)
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

    def _repeat_leading(self, tensor, copies):
        if tensor is None:
            return None
        return tensor.repeat(copies, *([1] * (tensor.dim() - 1)))

    def _combine_cfg_prediction(self, total_input, side_info, step_index, timestep_emb, size_emb, text_pack, guide_weight):
        batch = side_info.shape[0]

        def double(value):
            if isinstance(value, (list, tuple)):
                return [self._repeat_leading(item, 2) for item in value]
            return self._repeat_leading(value, 2)

        inject = torch.cat(
            [
                torch.ones((batch,), device=side_info.device),
                torch.zeros((batch,), device=side_info.device),
            ],
            dim=0,
        )
        doubled_pack = {
            "hidden": double(text_pack["hidden"]),
            "token_mask": double(text_pack["token_mask"]),
            "source_ids": text_pack["source_ids"],
            "source_mask": text_pack["source_mask"],
        }
        predicted, similarity = self._run_diffusion_model(
            double(total_input),
            double(side_info),
            torch.tensor([step_index], device=side_info.device),
            inject,
            timestep_emb=double(timestep_emb),
            size_emb=double(size_emb),
            context=None,
            text_pack=doubled_pack,
            return_text_stats=True,
        )
        pred_cond = predicted[:batch]
        pred_null = predicted[batch:]
        sim_cond = similarity[:batch] if similarity is not None else predicted.new_zeros((batch,))
        gate = self._text_gate_value(sim_cond, text_pack.get("source_mask"))
        gate = gate.view(batch, 1, 1).to(dtype=predicted.dtype)
        return pred_null + float(guide_weight) * gate * (pred_cond - pred_null)

    def _build_ddim_schedule(self):
        """Sub-schedule of diffusion indices. Values are the τ used by the DDIM update."""
        if self.sample_method == 'linear':
            stride = self.num_steps // self.sample_steps
            time_steps = np.asarray(list(range(0, self.num_steps, stride)))
        elif self.sample_method == "quad":
            time_steps = (np.linspace(0, np.sqrt(self.num_steps * 0.8), self.sample_steps) ** 2).astype(int)
        else:
            raise NotImplementedError(f"sampling method {self.sample_method} is not implemented!")
        time_steps = time_steps + 1
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])
        return time_steps, time_steps_prev

    def _denoiser_step_index(self, loop_index, time_steps):
        """Diffusion-step argument for the denoiser.

        Legacy DDIM passes the loop index 0..S-1. With sampler_fix it passes τ,
        the schedule entry whose ᾱ is the noise level of this reverse step.
        """
        if self.ddim and self.sampler_fix:
            return int(time_steps[loop_index])
        return int(loop_index)

    def _ddim_x0_direction_coeff(self, alpha_prev, alpha_tau):
        """Coefficient of x_τ in the η=0 x0 DDIM update.

        Song et al. 2021 eq. 12 with η=0 is
        sqrt((1-ᾱ_prev)/(1-ᾱ_τ)). The legacy expression omits the numerator square root.
        """
        alpha_prev = float(alpha_prev)
        alpha_tau = float(alpha_tau)
        if self.sampler_fix:
            return ((1.0 - alpha_prev) / (1.0 - alpha_tau)) ** 0.5
        return (1.0 - alpha_prev) / (1.0 - alpha_tau) ** 0.5

    def impute(self, observed_data, cond_mask, side_info, n_samples, guide_w=None, timesteps=None, timestep_emb=None, size_emb=None, context=None, pattern_text_pooled=None, pattern_text_mask=None, pattern_text_evidence_vec=None, controller_state=None, text_pack=None):
        B, K, L = observed_data.shape
        if self.ddim:
            time_steps, time_steps_prev = self._build_ddim_schedule()
        else:
            self.sample_steps = self.num_steps
            time_steps = None
            time_steps_prev = None
        if not self.noise_esti:
            cond_count = torch.sum(cond_mask, dim=2, keepdim=True)
            safe_count = cond_count.clamp(min=1.0)
            means = torch.sum(observed_data * cond_mask, dim=2, keepdim=True) / safe_count
            var_denom = (cond_count - 1.0).clamp(min=1.0)
            stdev = torch.sqrt(torch.sum((observed_data - means) ** 2 * cond_mask, dim=2, keepdim=True) / var_denom + 1e-5)
            observed_data = (observed_data - means) / stdev

        pattern_baseline = None
        diffusion_observed_data = observed_data
        if self.pattern_residual_diffusion and not self.noise_esti:
            if self.pattern_router_mode == "legacy" and self.baseline_form not in ("naive", "dlinear"):
                pattern_guidance_scale = self.guide_w_default if guide_w is None else float(guide_w)
            else:
                pattern_guidance_scale = 1.0
            pattern = self._compute_pattern_outputs(
                observed_data,
                cond_mask,
                text_pooled=pattern_text_pooled,
                text_mask=pattern_text_mask,
                text_evidence_vec=pattern_text_evidence_vec,
                guidance_scale=pattern_guidance_scale,
                controller_state=controller_state,
            )
            if self.final_method and self.pattern_router_mode == "legacy" and self.baseline_form not in ("naive", "dlinear"):
                pattern_baseline = self._compute_weak_pattern_baseline(
                    pattern,
                    observed_data,
                    torch.zeros_like(cond_mask),
                    use_training_target=False,
                )
            else:
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
                denoise_t = self._denoiser_step_index(t, time_steps)
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
                        use_cfg = self.text_injection == "gated" and self.cfg_output_combine and text_pack is not None
                        if use_cfg:
                            guide_weight = self.guide_w_default if guide_w is None else float(guide_w)
                            predicted = self._combine_cfg_prediction(
                                [res_input, moving_mean_input], side_info, denoise_t, timestep_emb, size_emb, text_pack, guide_weight
                            )
                        else:
                            predicted_seasonal = self._unwrap_diffmodel_output(
                                self.diffmodel_sesonal(res_input, side_info, torch.tensor([denoise_t]).to(self.device), cfg_mask, timestep_emb, size_emb, context)
                            ) # (2*B, K, L)
                            predicted_trend = self._unwrap_diffmodel_output(
                                self.diffmodel_trend(moving_mean_input, side_info, torch.tensor([denoise_t]).to(self.device), cfg_mask, timestep_emb, size_emb, context)
                            ) # (2*B, K, L)
                            predicted = predicted_seasonal + predicted_trend # (2*B, K, L)
                    else:
                        cond_obs = (cond_mask * diffusion_observed_data).unsqueeze(1) # (B, 1, K, L)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1) # (B, 1, K, L)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)
                        use_cfg = self.text_injection == "gated" and self.cfg_output_combine and text_pack is not None
                        if use_cfg:
                            guide_weight = self.guide_w_default if guide_w is None else float(guide_w)
                            predicted = self._combine_cfg_prediction(
                                diff_input, side_info, denoise_t, timestep_emb, size_emb, text_pack, guide_weight
                            )
                        elif self.save_attn:
                            predicted, attn = self.diffmodel(diff_input, side_info, torch.tensor([denoise_t]).to(self.device), cfg_mask, timestep_emb, size_emb, context) # (2*B, K, L)
                        else:
                            predicted = self.diffmodel(diff_input, side_info, torch.tensor([denoise_t]).to(self.device), cfg_mask, timestep_emb, size_emb, context) # (2*B, K, L)

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
                        # Legacy stops at x_1: the mean update requires t>1 and noise requires t>2,
                        # so the t=1 and t=0 denoiser calls are discarded. sampler_fix runs the
                        # posterior down through t=1 (deterministic x_0 mean, noise only for t>1)
                        # and returns the t=0 x_0 prediction.
                        if self.sampler_fix:
                            if t == 0:
                                current_sample = predicted
                            else:
                                coeff1 = (self.alpha_hat[t] ** 0.5 * (1 - self.alpha[t-1])) / (1 - self.alpha[t])
                                coeff2 = (self.alpha[t-1] ** 0.5 * self.beta[t]) / (1 - self.alpha[t])
                                current_sample = coeff1 * current_sample + coeff2 * predicted
                                if t > 1:
                                    noise = torch.randn_like(current_sample)
                                    sigma = (
                                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                                    ) ** 0.5
                                    current_sample += sigma * noise
                        elif t > 1:
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
                        aaa_ = self._ddim_x0_direction_coeff(self.alpha[tau_prev], self.alpha[tau])
                        current_sample = (
                            aaa_ * current_sample +
                            ((self.alpha[tau_prev])**0.5 - (self.alpha[tau])**0.5 * aaa_) * predicted
                        )

            sample = current_sample.detach()
            if pattern_baseline is not None:
                sample = sample + pattern_baseline.detach()
            imputed_samples[:, i] = sample
            if self.timestep_outer_mix != "none" and self.timestep_branch and timesteps is not None:
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
        text_ret_mask = batch.get("text_ret_mark")
        if text_ret_mask is not None:
            text_ret_mask = text_ret_mask.to(self.device).float().reshape(-1).clamp(0.0, 1.0)
        else:
            text_ret_mask = text_mask.clone()
        text_cot_mask = batch.get("text_cot_mark")
        if text_cot_mask is not None:
            text_cot_mask = text_cot_mask.to(self.device).float().reshape(-1).clamp(0.0, 1.0)
        else:
            text_cot_mask = text_mask.clone()
        text_raw = batch.get("text_raw", batch.get("texts"))
        text_ret = batch.get("text_ret", batch.get("retrieved_text"))
        text_cot = batch.get("text_cot", batch.get("cot_text"))
        text_raw_mark = batch.get("text_raw_mark")
        if text_raw_mark is not None and torch.is_tensor(text_raw_mark):
            text_raw_mark = text_raw_mark.to(self.device).float().reshape(-1).clamp(0.0, 1.0)
        elif text_raw_mark is not None:
            text_raw_mark = torch.tensor(text_raw_mark, device=self.device, dtype=torch.float32).reshape(-1).clamp(0.0, 1.0)
        else:
            text_raw_mark = self._mark_from_text_batch(text_raw, batch_size, self.device)
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
        target_feature_index = batch.get("target_feature_index")
        if target_feature_index is not None:
            target_feature_index = target_feature_index.to(self.device).long().reshape(-1)
        else:
            target_feature_index = torch.full(
                (observed_data.shape[0],),
                int(self.target_feature_index),
                device=self.device,
                dtype=torch.long,
            )

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
            text_ret_mask,
            text_cot_mask,
            trend_prior_num,
            trend_prior_text,
            text_evidence_vec,
            text_event_texts,
            text_event_source_ids,
            text_event_time_deltas,
            text_event_quality_feats,
            text_event_mask,
            target_feature_index,
            text_raw,
            text_ret,
            text_cot,
            text_raw_mark,
        )        

    def _unpack_forecasting_batch(self, data):
        defaults = {
            "feature_id": None,
            "timesteps": None,
            "texts": None,
            "text_mask": None,
            "text_ret_mask": None,
            "text_cot_mask": None,
            "trend_prior_num": None,
            "trend_prior_text": None,
            "text_evidence_vec": None,
            "text_event_texts": None,
            "text_event_source_ids": None,
            "text_event_time_deltas": None,
            "text_event_quality_feats": None,
            "text_event_mask": None,
            "target_feature_index": None,
            "text_raw": None,
            "text_ret": None,
            "text_cot": None,
            "text_raw_mark": None,
        }
        if len(data) >= 25:
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
                defaults["text_ret_mask"],
                defaults["text_cot_mask"],
                defaults["trend_prior_num"],
                defaults["trend_prior_text"],
                defaults["text_evidence_vec"],
                defaults["text_event_texts"],
                defaults["text_event_source_ids"],
                defaults["text_event_time_deltas"],
                defaults["text_event_quality_feats"],
                defaults["text_event_mask"],
                defaults["target_feature_index"],
                defaults["text_raw"],
                defaults["text_ret"],
                defaults["text_cot"],
                defaults["text_raw_mark"],
            ) = data
        elif len(data) >= 21:
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
                defaults["text_ret_mask"],
                defaults["text_cot_mask"],
                defaults["trend_prior_num"],
                defaults["trend_prior_text"],
                defaults["text_evidence_vec"],
                defaults["text_event_texts"],
                defaults["text_event_source_ids"],
                defaults["text_event_time_deltas"],
                defaults["text_event_quality_feats"],
                defaults["text_event_mask"],
                defaults["target_feature_index"],
            ) = data
        elif len(data) >= 18:
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

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask, target_feature_index=None):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        extracted_target_feature_index = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            if target_feature_index is not None:
                target_idx = int(target_feature_index[k].item()) if torch.is_tensor(target_feature_index) else int(target_feature_index)
                remaining = ind[ind != target_idx]
                np.random.shuffle(remaining)
                chosen = np.concatenate([[target_idx], remaining[: max(size - 1, 0)]])
                new_target_idx = 0
            else:
                np.random.shuffle(ind)
                chosen = ind[:size]
                new_target_idx = 0
            extracted_data.append(observed_data[k,chosen])
            extracted_mask.append(observed_mask[k,chosen])
            extracted_feature_id.append(feature_id[k,chosen])
            extracted_gt_mask.append(gt_mask[k,chosen])
            extracted_target_feature_index.append(new_target_idx)
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        extracted_target_feature_index = torch.as_tensor(extracted_target_feature_index, device=observed_data.device, dtype=torch.long)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask, extracted_target_feature_index
    
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

    def _mark_from_text_batch(self, texts, batch_size, device):
        marks = []
        items = texts if isinstance(texts, (list, tuple)) else [texts] * batch_size
        for item in list(items)[:batch_size]:
            normalized = "" if item is None else str(item).strip()
            marks.append(0.0 if normalized == "" or normalized.upper() == "NA" else 1.0)
        if len(marks) < batch_size:
            marks.extend([0.0] * (batch_size - len(marks)))
        return torch.tensor(marks, device=device, dtype=torch.float32)

    def _maybe_shuffle_text_alignment(self, bundle):
        if not self.shuffle_text_in_batch:
            return bundle
        batch = None
        for value in bundle.values():
            if torch.is_tensor(value) and value.dim() >= 1:
                batch = value.shape[0]
                break
            if isinstance(value, (list, tuple)):
                batch = len(value)
                break
        if batch is None or batch <= 1:
            return bundle
        perm = torch.randperm(batch)
        shuffled = {}
        for key, value in bundle.items():
            if torch.is_tensor(value) and value.shape[0] == batch:
                shuffled[key] = value[perm]
            elif isinstance(value, (list, tuple)) and len(value) == batch:
                order = perm.tolist()
                shuffled[key] = [value[index] for index in order]
            else:
                shuffled[key] = value
        return shuffled

    def _stable_token_id(self, token):
        digest = 0
        for char in str(token):
            digest = (digest * 131 + ord(char)) % 30000
        return int(digest)

    def _hash_token_vector(self, token_id, width):
        positions = torch.arange(width, dtype=torch.float32)
        angles = (float(token_id) + 1.0) * torch.exp(-positions / max(width / 8.0, 1.0))
        vector = torch.sin(angles)
        vector[1::2] = torch.cos(angles[1::2])
        return vector

    def _hash_encode_texts(self, texts, max_length, truncation_side):
        max_length = int(max_length)
        hidden = torch.zeros((len(texts), max_length, self.text_hidden_dim), device=self.device)
        attention = torch.zeros((len(texts), max_length), device=self.device)
        input_ids = torch.zeros((len(texts), max_length), device=self.device, dtype=torch.long)
        for row, text in enumerate(texts):
            tokens = str(text).split()
            budget = max(max_length - 2, 1)
            if len(tokens) > budget:
                tokens = tokens[-budget:] if truncation_side == "left" else tokens[:budget]
            ids = [101] + [1 + (self._stable_token_id(token) % 30000) for token in tokens] + [102]
            ids = ids[:max_length]
            for col, token_id in enumerate(ids):
                hidden[row, col] = self._hash_token_vector(token_id, self.text_hidden_dim).to(self.device)
                attention[row, col] = 1.0
                input_ids[row, col] = int(token_id)
        pooled = (hidden * attention.unsqueeze(-1)).sum(dim=1) / attention.sum(dim=1, keepdim=True).clamp(min=1.0)
        return hidden, pooled, {"input_ids": input_ids, "attention_mask": attention}

    def _encode_text_tokens(self, texts, max_length=None, truncation_side="right"):
        max_length = int(max_length or self.text_max_length)
        if not self.text_encoder_eval:
            return self._encode_text_tokens_impl(texts, max_length, truncation_side)
        texts = [str(item) if item is not None else "NA" for item in texts]
        hidden_rows = [None] * len(texts)
        pooled_rows = [None] * len(texts)
        mask_rows = [None] * len(texts)
        missing_text = []
        missing_at = []
        for index, text in enumerate(texts):
            key = (str(truncation_side), max_length, text)
            cached = self._text_encode_cache.get(key)
            if cached is None:
                if text not in {item for item in missing_text}:
                    missing_text.append(text)
                missing_at.append(index)
                continue
            hidden_rows[index], pooled_rows[index], mask_rows[index] = cached
        if missing_text:
            hidden, pooled, mask = self._encode_text_tokens_impl(missing_text, max_length, truncation_side)
            located = {text: row for row, text in enumerate(missing_text)}
            for index in missing_at:
                text = texts[index]
                row = located[text]
                stored = (
                    hidden[row].detach().to("cpu"),
                    pooled[row].detach().to("cpu"),
                    mask[row].detach().to("cpu"),
                )
                self._text_encode_cache[(str(truncation_side), max_length, text)] = stored
                hidden_rows[index], pooled_rows[index], mask_rows[index] = stored
        hidden = torch.stack([row.to(self.device) for row in hidden_rows], dim=0)
        pooled = torch.stack([row.to(self.device) for row in pooled_rows], dim=0)
        mask = torch.stack([row.to(self.device) for row in mask_rows], dim=0)
        return hidden, pooled, mask

    def _encode_text_tokens_impl(self, texts, max_length, truncation_side):
        if self.tokenizer is None:
            hidden, pooled, token_input = self._hash_encode_texts(texts, max_length, truncation_side)
            return hidden, pooled, token_input["attention_mask"]
        previous_side = getattr(self.tokenizer, "truncation_side", "right")
        self.tokenizer.truncation_side = truncation_side
        try:
            hidden, pooled, token_input = self._encode_text_source_with_options(
                texts, max_length=max_length, return_sequence=True
            )
        finally:
            self.tokenizer.truncation_side = previous_side
        return hidden, pooled, token_input["attention_mask"]

    def _build_text_pack(self, text_raw, text_ret, text_cot, raw_mark, ret_mark, cot_mark, router_text_mask):
        batch = int(raw_mark.shape[0])
        sources = [
            ("raw", text_raw if text_raw is not None else ["NA"] * batch, "left", raw_mark),
            ("ret", text_ret if text_ret is not None else ["NA"] * batch, "right", ret_mark),
        ]
        if self.cot_as_source:
            sources.append(("cot", text_cot if text_cot is not None else ["NA"] * batch, "right", cot_mark))
        hiddens = []
        token_masks = []
        pooled = []
        source_ids = []
        source_marks = []
        for source_index, (_, texts, side, mark) in enumerate(sources):
            if not isinstance(texts, (list, tuple)):
                texts = [texts] * batch
            hidden, pooled_source, attention = self._encode_text_tokens(list(texts), truncation_side=side)
            mark = mark.reshape(batch).to(device=hidden.device, dtype=hidden.dtype)
            token_mask = attention.to(device=hidden.device, dtype=hidden.dtype) * mark.unsqueeze(1)
            hiddens.append(hidden)
            token_masks.append(token_mask)
            pooled.append(pooled_source)
            source_ids.append(torch.full((hidden.shape[1],), source_index, device=hidden.device, dtype=torch.long))
            source_marks.append(mark)
        source_mask = torch.stack(source_marks, dim=1)
        if source_mask.shape[1] < self.n_text_sources:
            source_mask = F.pad(source_mask, (0, self.n_text_sources - source_mask.shape[1]))
        pooled_tensor = torch.stack(pooled, dim=1)
        weights = source_mask[:, : pooled_tensor.shape[1]].unsqueeze(-1)
        router_pooled = (pooled_tensor * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        any_text = (source_mask.sum(dim=1) > 0).to(dtype=router_pooled.dtype)
        if router_text_mask is not None:
            any_text = any_text * router_text_mask.reshape(-1).to(dtype=any_text.dtype)
        text_pack = {
            "hidden": torch.cat(hiddens, dim=1),
            "token_mask": torch.cat(token_masks, dim=1),
            "source_ids": torch.cat(source_ids, dim=0),
            "source_mask": source_mask[:, : self.n_text_sources],
        }
        return text_pack, router_pooled, any_text

    def _encode_text_source(self, text):
        return self._encode_text_source_with_options(text, max_length=self.text_max_length, return_sequence=False)

    def _encode_text_source_with_options(self, text, max_length=None, return_sequence=True):
        if self.tokenizer is None:
            texts = [text] if isinstance(text, str) else [str(item) if item is not None else "NA" for item in text]
            hidden, pooled, token_input = self._hash_encode_texts(texts, int(max_length or self.text_max_length), "right")
            return (hidden if return_sequence else None), pooled, token_input
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

    def _build_forecast_policy_state(self, observed_data, cond_mask, trend_prior, text_mask, retrieval_mask, reasoning_mask, text_evidence_vec=None):
        if self.forecast_policy_controller is None or trend_prior is None:
            return None
        history_len = min(max(int(self.lookback_len), 1), observed_data.shape[-1])
        if observed_data.shape[1] > 1:
            target_idx = int(min(max(self.target_feature_index, 0), observed_data.shape[1] - 1))
            history = observed_data[:, target_idx:target_idx + 1, :history_len]
        else:
            history = observed_data[:, :, :history_len]
        prior = trend_prior.mean(dim=1) if trend_prior.dim() == 3 else trend_prior
        source_utilities = self._compute_source_utilities(
            observed_data,
            cond_mask,
            trend_prior=trend_prior,
            text_mask=text_mask,
            retrieval_mask=retrieval_mask,
            reasoning_mask=reasoning_mask,
            text_evidence_vec=text_evidence_vec,
        )
        return self.forecast_policy_controller(
            history,
            prior,
            text_mask=text_mask,
            retrieval_mask=retrieval_mask,
            reasoning_mask=reasoning_mask,
            source_utilities=source_utilities,
        )

    def _apply_covariate_gate(self, observed_data, observed_mask, gt_mask, controller_state):
        if controller_state is None or observed_data.shape[1] <= 1:
            return observed_data, observed_mask, gt_mask
        covariate_gate = controller_state.get("covariate_gate")
        if covariate_gate is None:
            return observed_data, observed_mask, gt_mask
        target_idx = int(min(max(self.target_feature_index, 0), observed_data.shape[1] - 1))
        aux_indices = [idx for idx in range(observed_data.shape[1]) if idx != target_idx]
        if not aux_indices:
            return observed_data, observed_mask, gt_mask
        gate = covariate_gate.to(device=observed_data.device, dtype=observed_data.dtype).reshape(-1, 1, 1)
        history_slice = slice(0, min(self.lookback_len, observed_data.shape[-1]))
        observed_data = observed_data.clone()
        observed_mask = observed_mask.clone()
        gt_mask = gt_mask.clone()
        observed_data[:, aux_indices, history_slice] = observed_data[:, aux_indices, history_slice] * gate
        observed_mask[:, aux_indices, history_slice] = observed_mask[:, aux_indices, history_slice] * gate
        gt_mask[:, aux_indices, history_slice] = gt_mask[:, aux_indices, history_slice] * gate
        return observed_data, observed_mask, gt_mask

    def _fill_history_for_aux(self, observed_data, cond_mask):
        history = observed_data[:, :, : self.lookback_len]
        history_mask = cond_mask[:, :, : self.lookback_len].float()
        denom = history_mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        hist_mean = (history * history_mask).sum(dim=2, keepdim=True) / denom
        return torch.where(history_mask > 0, history, hist_mean.expand_as(history))

    def _build_policy_condition_vector(self, trend_prior, controller_state, text_mask, text_evidence_vec):
        if trend_prior is None:
            batch_size = text_mask.shape[0] if text_mask is not None else 1
            trend = torch.zeros((batch_size, 3), device=self.device)
        else:
            trend = trend_prior.mean(dim=1) if trend_prior.dim() == 3 else trend_prior
        batch_size = trend.shape[0]
        device = trend.device
        dtype = trend.dtype

        if controller_state is None:
            regime_probs = torch.zeros((batch_size, 5), device=device, dtype=dtype)
            uncertainty_score = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            clarity_score = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            guidance_scale = torch.ones((batch_size, 1), device=device, dtype=dtype)
            residual_scale = torch.ones((batch_size, 1), device=device, dtype=dtype)
            sample_budget_scale = torch.ones((batch_size, 1), device=device, dtype=dtype)
            evidence_conflict = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        else:
            regime_probs = controller_state.get("regime_probs", torch.zeros((batch_size, 5), device=device, dtype=dtype)).to(device=device, dtype=dtype)
            uncertainty_score = controller_state.get("uncertainty_score", torch.zeros((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            clarity_score = controller_state.get("clarity_score", torch.zeros((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            guidance_scale = controller_state.get("guidance_scale", torch.ones((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            residual_scale = controller_state.get("residual_scale", torch.ones((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            sample_budget_scale = controller_state.get("sample_budget_scale", torch.ones((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            evidence_conflict = controller_state.get("evidence_conflict", torch.zeros((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
        if controller_state is None:
            covariate_gate = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            coarse_gate = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            uncertainty_gate = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            text_gate = guidance_scale
            seasonality_score = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            source_text = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            source_cov = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            source_coarse = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            source_unc = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        else:
            covariate_gate = controller_state.get("covariate_gate", torch.zeros((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            coarse_gate = controller_state.get("coarse_gate", torch.zeros((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            uncertainty_gate = controller_state.get("uncertainty_gate", torch.zeros((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            text_gate = controller_state.get("text_gate", guidance_scale.reshape(-1)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            seasonality_score = controller_state.get("seasonality_score", torch.zeros((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            source_text = controller_state.get("source_utilities", {}).get("text", torch.zeros((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            source_cov = controller_state.get("source_utilities", {}).get("covariate", torch.zeros((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            source_coarse = controller_state.get("source_utilities", {}).get("coarse", torch.zeros((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)
            source_unc = controller_state.get("source_utilities", {}).get("uncertainty", torch.zeros((batch_size,), device=device, dtype=dtype)).reshape(batch_size, 1).to(device=device, dtype=dtype)

        if text_mask is None:
            text_availability = torch.ones((batch_size, 1), device=device, dtype=dtype)
        else:
            text_availability = text_mask.reshape(batch_size, 1).to(device=device, dtype=dtype)
        if text_evidence_vec is not None:
            evidence = text_evidence_vec.to(device=device, dtype=dtype)
            if evidence.dim() == 1:
                evidence = evidence.reshape(batch_size, -1)
            text_quality = evidence[:, :1]
        else:
            text_quality = text_availability

        cond_vec = torch.cat(
            [
                trend.to(device=device, dtype=dtype),
                regime_probs,
                uncertainty_score,
                clarity_score,
                guidance_scale,
                residual_scale,
                sample_budget_scale,
                evidence_conflict,
                covariate_gate,
                coarse_gate,
                uncertainty_gate,
                text_gate,
                seasonality_score,
                source_text,
                source_cov,
                source_coarse,
                source_unc,
                text_quality,
                text_availability,
            ],
            dim=1,
        )
        if cond_vec.shape[1] < self.policy_cond_dim:
            cond_vec = F.pad(cond_vec, (0, self.policy_cond_dim - cond_vec.shape[1]))
        elif cond_vec.shape[1] > self.policy_cond_dim:
            cond_vec = cond_vec[:, : self.policy_cond_dim]
        return cond_vec

    def _compute_aux_head_outputs(self, observed_data, cond_mask, trend_prior=None, controller_state=None, text_mask=None, text_evidence_vec=None):
        if (self.coarse_forecast_head is None) and (self.uncertainty_head is None):
            return None
        history = self._fill_history_for_aux(observed_data, cond_mask)
        cond_vec = self._build_policy_condition_vector(trend_prior, controller_state, text_mask, text_evidence_vec)
        outputs = {"cond_vec": cond_vec}
        if self.coarse_forecast_head is not None:
            outputs["coarse_future"] = self.coarse_forecast_head(history, cond_vec=cond_vec)
        if self.uncertainty_head is not None:
            loc, logvar = self.uncertainty_head(history, cond_vec=cond_vec)
            outputs["uncertainty_loc"] = loc
            outputs["uncertainty_logvar"] = logvar
        return outputs

    def _build_aux_future_baseline(self, aux_outputs, controller_state, device, dtype, total_length):
        if aux_outputs is None or "coarse_future" not in aux_outputs or self.coarse_forecast_blend <= 0:
            return None
        coarse_future = aux_outputs["coarse_future"].to(device=device, dtype=dtype)
        batch_size = coarse_future.shape[0]
        future_weight = torch.ones((batch_size, self.pred_len), device=device, dtype=dtype)
        if controller_state is not None and controller_state.get("fine_mask") is not None:
            fine_mask = controller_state["fine_mask"].to(device=device, dtype=dtype)
            future_weight = 1.0 - 0.5 * fine_mask
        coarse_future = coarse_future * future_weight.unsqueeze(1)
        if "uncertainty_logvar" in aux_outputs:
            confidence = torch.exp(-0.5 * aux_outputs["uncertainty_logvar"].to(device=device, dtype=dtype)).clamp(min=0.25, max=2.0)
            coarse_future = coarse_future * confidence
        if controller_state is not None and controller_state.get("coarse_gate") is not None:
            coarse_gate = controller_state["coarse_gate"].to(device=device, dtype=dtype).reshape(batch_size, 1, 1)
            coarse_future = coarse_future * coarse_gate
        aux_baseline = torch.zeros((batch_size, coarse_future.shape[1], total_length), device=device, dtype=dtype)
        aux_baseline[:, :, self.lookback_len:self.lookback_len + self.pred_len] = coarse_future
        return self.coarse_forecast_blend * aux_baseline

    def _calc_coarse_aux_loss(self, aux_outputs, observed_data, target_mask, controller_state=None):
        if aux_outputs is None or "coarse_future" not in aux_outputs or self.coarse_forecast_weight <= 0:
            return torch.zeros((), device=observed_data.device)
        future_target = observed_data[:, :, self.lookback_len:self.lookback_len + self.pred_len]
        future_mask = target_mask[:, :, self.lookback_len:self.lookback_len + self.pred_len].float()
        if future_mask.sum() <= 0:
            return torch.zeros((), device=observed_data.device)
        weights = torch.ones_like(future_mask)
        if controller_state is not None and controller_state.get("risk_scores") is not None:
            risk = controller_state["risk_scores"].to(device=observed_data.device, dtype=observed_data.dtype)
            weights = weights * (1.0 + 0.5 * risk.unsqueeze(1))
        residual = aux_outputs["coarse_future"].to(device=observed_data.device, dtype=observed_data.dtype) - future_target
        delta = float(max(self.multi_res_huber_delta, 1e-6))
        abs_res = residual.abs()
        huber = torch.where(abs_res <= delta, 0.5 * residual ** 2, delta * abs_res - 0.5 * (delta ** 2))
        weighted_mask = future_mask * weights
        denom = weighted_mask.sum().clamp(min=1.0)
        loss = (huber * weighted_mask).sum() / denom
        if controller_state is not None and controller_state.get("coarse_gate") is not None:
            loss = loss * controller_state["coarse_gate"].to(device=observed_data.device, dtype=observed_data.dtype).mean()
        return loss

    def _calc_uncertainty_aux_loss(self, aux_outputs, observed_data, target_mask, controller_state=None):
        if aux_outputs is None or "uncertainty_loc" not in aux_outputs or self.uncertainty_forecast_weight <= 0:
            return torch.zeros((), device=observed_data.device)
        future_target = observed_data[:, :, self.lookback_len:self.lookback_len + self.pred_len]
        future_mask = target_mask[:, :, self.lookback_len:self.lookback_len + self.pred_len].float()
        if future_mask.sum() <= 0:
            return torch.zeros((), device=observed_data.device)
        loc = aux_outputs["uncertainty_loc"].to(device=observed_data.device, dtype=observed_data.dtype)
        logvar = aux_outputs["uncertainty_logvar"].to(device=observed_data.device, dtype=observed_data.dtype)
        precision = torch.exp(-logvar)
        nll = 0.5 * (precision * (future_target - loc) ** 2 + logvar)
        weights = torch.ones_like(future_mask)
        if controller_state is not None and controller_state.get("risk_scores") is not None:
            risk = controller_state["risk_scores"].to(device=observed_data.device, dtype=observed_data.dtype)
            weights = weights * (1.0 + 0.5 * risk.unsqueeze(1))
        weighted_mask = future_mask * weights
        denom = weighted_mask.sum().clamp(min=1.0)
        loss = (nll * weighted_mask).sum() / denom
        if controller_state is not None and controller_state.get("uncertainty_gate") is not None:
            loss = loss * controller_state["uncertainty_gate"].to(device=observed_data.device, dtype=observed_data.dtype).mean()
        return loss

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
        text_ret_mask = unpacked["text_ret_mask"]
        text_cot_mask = unpacked["text_cot_mask"]
        trend_prior_num = unpacked["trend_prior_num"]
        trend_prior_text = unpacked["trend_prior_text"]
        text_evidence_vec = unpacked["text_evidence_vec"]
        text_event_texts = unpacked["text_event_texts"]
        text_event_source_ids = unpacked["text_event_source_ids"]
        text_event_time_deltas = unpacked["text_event_time_deltas"]
        text_event_quality_feats = unpacked["text_event_quality_feats"]
        text_event_mask = unpacked["text_event_mask"]
        target_feature_index = unpacked["target_feature_index"]
        text_raw = unpacked["text_raw"]
        text_ret = unpacked["text_ret"]
        text_cot = unpacked["text_cot"]
        text_raw_mark = unpacked["text_raw_mark"]
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask,target_feature_index = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask,target_feature_index=target_feature_index)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if target_feature_index is not None:
            self.target_feature_index = int(target_feature_index[0].item())

        text_pack = None
        if self.with_texts and (self.text_injection == "gated" or self.pattern_router_mode == "linear_history_text"):
            if text_raw_mark is None:
                text_raw_mark = self._mark_from_text_batch(text_raw if text_raw is not None else texts, observed_data.shape[0], self.device)
            shuffled = self._maybe_shuffle_text_alignment({
                "text_raw": text_raw if text_raw is not None else texts,
                "text_ret": text_ret,
                "text_cot": text_cot,
                "text_raw_mark": text_raw_mark,
                "text_ret_mask": text_ret_mask,
                "text_cot_mask": text_cot_mask,
            })
            text_pack, text_pooled, any_text = self._build_text_pack(
                shuffled["text_raw"],
                shuffled["text_ret"],
                shuffled["text_cot"],
                shuffled["text_raw_mark"],
                shuffled["text_ret_mask"] if shuffled["text_ret_mask"] is not None else text_raw_mark,
                shuffled["text_cot_mask"] if shuffled["text_cot_mask"] is not None else text_raw_mark,
                text_mask,
            )
            text_mask = any_text
            if self.text_injection != "gated":
                text_pack = None
        elif self.with_texts:
            _, text_pooled, _ = self._encode_text_source(texts)
        else:
            text_pooled = None
            if text_mask is not None:
                text_mask = torch.zeros_like(text_mask)
            if text_ret_mask is not None:
                text_ret_mask = torch.zeros_like(text_ret_mask)
            if text_cot_mask is not None:
                text_cot_mask = torch.zeros_like(text_cot_mask)
            text_evidence_vec = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )
        controller_state = self._build_forecast_policy_state(
            observed_data,
            cond_mask,
            trend_prior_text,
            text_mask,
            text_ret_mask,
            text_cot_mask,
            text_evidence_vec=text_evidence_vec,
        )
        observed_data, observed_mask, gt_mask = self._apply_covariate_gate(observed_data, observed_mask, gt_mask, controller_state)
        if is_train == 0:
            cond_mask = gt_mask
        else:
            cond_mask = self.get_test_pattern_mask(observed_mask, gt_mask)

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id, timesteps, texts)

        if self.timestep_emb_cat:
            timestep_emb = self.get_timestep_info(timesteps)
        else:
            timestep_emb = None

        if self.relative_size_emb_cat:
            size_emb = self.get_relative_size_info(observed_data)
        else:
            size_emb = None

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
            controller_state=controller_state,
            text_pack=text_pack,
        )

    def evaluate(self, batch, n_samples, guide_w):
        data = self.process_data(batch)
        observed_data, observed_mask, observed_tp, gt_mask, unpacked = self._unpack_forecasting_batch(data)
        feature_id = unpacked["feature_id"]
        timesteps = unpacked["timesteps"]
        texts = unpacked["texts"]
        text_mask = unpacked["text_mask"]
        text_ret_mask = unpacked["text_ret_mask"]
        text_cot_mask = unpacked["text_cot_mask"]
        trend_prior_num = unpacked["trend_prior_num"]
        trend_prior_text = unpacked["trend_prior_text"]
        text_evidence_vec = unpacked["text_evidence_vec"]
        text_event_texts = unpacked["text_event_texts"]
        text_event_source_ids = unpacked["text_event_source_ids"]
        text_event_time_deltas = unpacked["text_event_time_deltas"]
        text_event_quality_feats = unpacked["text_event_quality_feats"]
        text_event_mask = unpacked["text_event_mask"]
        target_feature_index = unpacked["target_feature_index"]
        text_raw = unpacked["text_raw"]
        text_ret = unpacked["text_ret"]
        text_cot = unpacked["text_cot"]
        text_raw_mark = unpacked["text_raw_mark"]

        with torch.no_grad():
            if target_feature_index is not None:
                self.target_feature_index = int(target_feature_index[0].item())
            cond_mask = gt_mask

            tokens = None
            text_pack = None
            if self.with_texts and (self.text_injection == "gated" or self.pattern_router_mode == "linear_history_text"):
                if text_raw_mark is None:
                    text_raw_mark = self._mark_from_text_batch(text_raw if text_raw is not None else texts, observed_data.shape[0], self.device)
                shuffled = self._maybe_shuffle_text_alignment({
                    "text_raw": text_raw if text_raw is not None else texts,
                    "text_ret": text_ret,
                    "text_cot": text_cot,
                    "text_raw_mark": text_raw_mark,
                    "text_ret_mask": text_ret_mask,
                    "text_cot_mask": text_cot_mask,
                })
                text_pack, text_pooled, any_text = self._build_text_pack(
                    shuffled["text_raw"],
                    shuffled["text_ret"],
                    shuffled["text_cot"],
                    shuffled["text_raw_mark"],
                    shuffled["text_ret_mask"] if shuffled["text_ret_mask"] is not None else text_raw_mark,
                    shuffled["text_cot_mask"] if shuffled["text_cot_mask"] is not None else text_raw_mark,
                    text_mask,
                )
                text_mask = any_text
                if self.text_injection != "gated":
                    text_pack = None
            elif self.with_texts:
                _, text_pooled, token_input = self._encode_text_source(texts)
                if self.save_token and self.tokenizer is not None:
                    tokens = self.tokenizer.batch_decode(token_input['input_ids'])
            else:
                text_pooled = None
                if text_mask is not None:
                    text_mask = torch.zeros_like(text_mask)
                if text_ret_mask is not None:
                    text_ret_mask = torch.zeros_like(text_ret_mask)
                if text_cot_mask is not None:
                    text_cot_mask = torch.zeros_like(text_cot_mask)
                text_evidence_vec = None
            controller_state = self._build_forecast_policy_state(
                observed_data,
                cond_mask,
                trend_prior_text,
                text_mask,
                text_ret_mask,
                text_cot_mask,
                text_evidence_vec=text_evidence_vec,
            )
            observed_data, observed_mask, gt_mask = self._apply_covariate_gate(observed_data, observed_mask, gt_mask, controller_state)
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
            if self.save_attn:
                samples, attn = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=None, pattern_text_pooled=text_pooled, pattern_text_mask=text_mask, pattern_text_evidence_vec=text_evidence_vec, controller_state=controller_state, text_pack=text_pack)
            else:
                samples = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=None, pattern_text_pooled=text_pooled, pattern_text_mask=text_mask, pattern_text_evidence_vec=text_evidence_vec, controller_state=controller_state, text_pack=text_pack)

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
