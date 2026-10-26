"""One synthetic train step and a short sample for decomp and non-decomp models.

Time-MMD CSVs and bert-base-uncased weights are not in this VM. Text is a
deterministic hash of the fake strings, not a frozen BERT.

The flags-off run must match a90347e. The flags-on run turns on the xattn
protocol switches. Cyclic text shift stays off, matching A1: it is the control,
not the main method. Validation still uses every diffusion index.
"""

import torch

from main_model import CSDI_Forecasting
from tests.synthetic_factory import make_batch, make_model, xattn_config


FLAGS_OFF = {
    "non-decomp": 2.740398,
    "decomp": 2.665316,
}


def trainable_count(model):
    return sum(
        param.numel()
        for name, param in model.named_parameters()
        if param.requires_grad and "text_encoder" not in name
    )


def protocol_model(decomp, layers=2, num_steps=3, channels=32, target_dim=2, lookback=12, pred=4):
    """Same synthetic net as make_model, with the A1 protocol switches on."""
    config = xattn_config(
        decomp=decomp,
        lookback=lookback,
        pred=pred,
        num_steps=num_steps,
        layers=layers,
        channels=channels,
    )
    config["train"].update({
        "eval_reseed": True,
        "save_window_metrics": True,
        "point_metric_pairing": True,
        "valid_t_schedule": "all",
        "named_output_dir": True,
    })
    config["diffusion"].update({
        "time_attn_dropout": 0.1,
        "ddim_x0_endpoint": True,
        "ddim_unique_steps": True,
    })
    config["model"].update({
        "ext_text_mode": "time_aligned",
        "ext_future_year_filter": True,
        "raw_placeholder_filter": True,
        "text_sources": "auto_from_train",
        "gate_beta": "auto_identifiable",
        "text_shift_mode": "none",
        "gate_constant_source": "last_epoch",
        "gate_ref_shared_dropout": True,
        "rag_query_unaugmented": True,
        "log_control_calls": True,
    })
    return CSDI_Forecasting(config, torch.device("cpu"), target_dim, [lookback, pred])


def run_once(name, decomp, protocol=False):
    torch.manual_seed(0)
    if protocol:
        model = protocol_model(decomp=decomp, layers=2, num_steps=3, channels=32, target_dim=2)
    else:
        model = make_model(decomp=decomp, layers=2, num_steps=3, channels=32, target_dim=2)
    model.train()
    batch = make_batch(model, batch_size=2, target_dim=2)
    loss = model(batch, is_train=1)
    loss.backward()
    alpha_grads = []
    denoisers = [model.diffmodel_sesonal, model.diffmodel_trend] if decomp else [model.diffmodel]
    for denoiser in denoisers:
        for layer in denoiser.residual_layers:
            grad = None if layer.text_alpha is None or layer.text_alpha.grad is None else float(layer.text_alpha.grad.abs())
            alpha_grads.append(grad)
    gate_grad = None if model.text_gate.a.grad is None else float(model.text_gate.a.grad.abs())
    model.eval()
    torch.manual_seed(1)
    samples, observed, target_mask, _, _ = model.evaluate(batch, n_samples=2, guide_w=1.0)
    loss_value = float(loss.detach())
    params = trainable_count(model)
    label = "protocol-on" if protocol else "flags-off"
    print(
        f"{label} {name}: loss={loss_value:.6f} "
        f"params={params} "
        f"alpha_abs_grad={alpha_grads} "
        f"gate_a_abs_grad={gate_grad} "
        f"samples={tuple(samples.shape)} "
        f"sample_mean={float(samples.mean()):.4f} "
        f"future_target_mean={float((observed * target_mask).sum() / target_mask.sum()):.4f}"
    )
    if not torch.isfinite(loss):
        raise SystemExit(f"{name} loss is not finite")
    if samples.shape != (2, 2, 2, model.lookback_len + model.pred_len):
        raise SystemExit(f"{name} unexpected sample shape {tuple(samples.shape)}")
    return loss_value, params


def main():
    print("synthetic smoke: no Time-MMD data, no BERT weights, llm=hash")
    off = {}
    on = {}
    for name, decomp in (("non-decomp", False), ("decomp", True)):
        off[name] = run_once(name, decomp=decomp, protocol=False)
        on[name] = run_once(name, decomp=decomp, protocol=True)
    for name, expected in FLAGS_OFF.items():
        if abs(off[name][0] - expected) > 5e-7:
            raise SystemExit(f"flags-off {name} loss {off[name][0]:.6f} != {expected:.6f}")
        if on[name][1] != off[name][1]:
            raise SystemExit(f"protocol-on {name} changed the parameter count")
    print(
        "flags-off matches a90347e; protocol-on losses "
        f"non-decomp={on['non-decomp'][0]:.6f} decomp={on['decomp'][0]:.6f}"
    )


if __name__ == "__main__":
    main()
