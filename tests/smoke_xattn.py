"""One synthetic train step and a short sample for decomp and non-decomp models.

Time-MMD CSVs and bert-base-uncased weights are not in this VM. Text is a
deterministic hash of the fake strings, not a frozen BERT.
"""

import torch

from tests.synthetic_factory import make_batch, make_model


def trainable_count(model):
    return sum(
        param.numel()
        for name, param in model.named_parameters()
        if param.requires_grad and "text_encoder" not in name
    )


def run_once(name, decomp):
    torch.manual_seed(0)
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
    print(
        f"{name}: loss={float(loss.detach()):.6f} "
        f"params={trainable_count(model)} "
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


def main():
    print("synthetic smoke: no Time-MMD data, no BERT weights, llm=hash")
    run_once("non-decomp", decomp=False)
    run_once("decomp", decomp=True)


if __name__ == "__main__":
    main()
