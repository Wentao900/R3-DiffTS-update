"""Small CSDI configs and batches for the gated cross-attention tests.

Time-MMD is not in this environment, and BERT weights are not downloaded.
These helpers build a hash-token model and fake text.
"""

import torch

from main_model import CSDI_Forecasting


def xattn_config(decomp=False, lookback=12, pred=4, num_steps=3, layers=2, channels=32):
    return {
        "train": {
            "epochs": 1,
            "multi_res_loss_weight": 0.0,
            "multi_res_horizons": [],
            "pattern_expert_weight": 0.05,
            "pattern_consistency_weight": 0.0,
            "auxiliary_loss_max_ratio": 0.0,
            "coarse_forecast_weight": 0.0,
            "uncertainty_forecast_weight": 0.0,
            "valid_loss_main_only": True,
        },
        "diffusion": {
            "layers": layers,
            "channels": channels,
            "nheads": 4,
            "diffusion_embedding_dim": 16,
            "beta_start": 0.0001,
            "beta_end": 0.2,
            "num_steps": num_steps,
            "schedule": "linear",
            "is_linear": False,
            "noise_esti": False,
            "cfg": True,
            "c_mask_prob": 0.0,
            "ddim": False,
            "sample_steps": num_steps,
            "sample_method": "linear",
            "pre_norm": False,
            "dropout": 0.0,
            "attn_drop": 0.0,
            "time_weight": 0.0,
            "mask_noisy_target": True,
        },
        "model": {
            "timeemb": 8,
            "featureemb": 4,
            "is_unconditional": 0,
            "target_strategy": "test",
            "timestep_branch": False,
            "timestep_emb_cat": True,
            "timestep_dim": 3,
            "timestep_outer_mix": "none",
            "with_texts": True,
            "relative_size_emb_cat": False,
            "decomp": bool(decomp),
            "llm": "hash",
            "llm_layers": 0,
            "context_dim": 32,
            "lookback_len": lookback,
            "pred_len": pred,
            "domain": "synthetic",
            "save_attn": False,
            "save_token": False,
            "num_sample_features": 8,
            "text_max_length": 12,
            "text_encode_batch_size": 4,
            "text_injection": "gated",
            "pattern_expert_set": "fpp4",
            "pattern_router_mode": "linear_history",
            "baseline_form": "fpp4",
            "season_period": 4,
            "text_validity_gate": True,
            "text_gate_mode": "learned",
            "text_validity_loss": True,
            "cfg_output_combine": True,
            "n_text_sources": 2,
            "cot_as_source": False,
            "force_legacy_entrypoint": False,
            "use_forecast_policy_controller": False,
            "use_aux_forecast_heads": False,
            "use_coarse_forecast_head": False,
            "use_uncertainty_head": False,
            "final_method": False,
            "guide_mode": "manual",
            "pattern_text_evidence": False,
            "pattern_reliability": False,
            "pattern_aux_reliability": False,
            "pattern_residual_diffusion": True,
        },
    }


def make_model(decomp=False, lookback=12, pred=4, num_steps=3, layers=2, channels=32, target_dim=2):
    config = xattn_config(
        decomp=decomp,
        lookback=lookback,
        pred=pred,
        num_steps=num_steps,
        layers=layers,
        channels=channels,
    )
    device = torch.device("cpu")
    model = CSDI_Forecasting(config, device, target_dim, [lookback, pred])
    return model


def make_batch(model, batch_size=2, target_dim=2, raw_texts=None, ret_texts=None):
    lookback = int(model.lookback_len)
    pred = int(model.pred_len)
    length = lookback + pred
    generator = torch.Generator().manual_seed(7)
    observed = torch.randn(batch_size, length, target_dim, generator=generator)
    observed_mask = torch.ones(batch_size, length, target_dim)
    gt_mask = torch.ones(batch_size, length, target_dim)
    gt_mask[:, lookback:, :] = 0.0
    timepoints = torch.arange(length, dtype=torch.float32).view(1, length).repeat(batch_size, 1)
    timestep_dim = model.timestep_emb[0].in_features
    timesteps = torch.randn(batch_size, length, timestep_dim, generator=generator)
    if raw_texts is None:
        raw_texts = [
            "old report older report newest raw report %d" % index
            for index in range(batch_size)
        ]
    if ret_texts is None:
        ret_texts = ["retrieved note %d about the series" % index for index in range(batch_size)]
    return {
        "observed_data": observed,
        "observed_mask": observed_mask,
        "gt_mask": gt_mask,
        "timepoints": timepoints,
        "timesteps": timesteps,
        "texts": list(raw_texts),
        "text_raw": list(raw_texts),
        "text_ret": list(ret_texts),
        "text_cot": ["NA"] * batch_size,
        "text_mark": torch.ones(batch_size),
        "text_raw_mark": torch.ones(batch_size),
        "text_ret_mark": torch.ones(batch_size),
        "text_cot_mark": torch.zeros(batch_size),
        "target_feature_index": torch.zeros(batch_size, dtype=torch.long),
    }
