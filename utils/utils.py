import numpy as np
import torch
from torch.optim import Adam, AdamW
from tqdm import tqdm
import os
import json
import math


FORECAST_CANDIDATE_NAMES = [
    "sample_mean",
    "sample_median",
    "last",
    "linear_trend",
    "history_mean",
    "mean_revert",
    "event_decay",
    "season_2",
    "season_3",
    "season_4",
    "season_6",
    "season_7",
    "season_12",
    "season_24",
]


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=float(config["lr"]), weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    warmup_epochs = max(int(config.get("lr_warmup_epochs", 0)), 0)
    max_grad_norm = float(config.get("max_grad_norm", 0.0))
    base_lr = float(config["lr"])
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    saved_best_model = False
    max_train_batches = min(len(train_loader), int(config["itr_per_epoch"]))
    print(f"[train] batches per epoch: {max_train_batches} (loader={len(train_loader)}, itr_per_epoch={config['itr_per_epoch']})")
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.current_epoch = epoch_no
        model.total_epochs = max(int(config["epochs"]), 1)
        if warmup_epochs > 0 and epoch_no < warmup_epochs:
            warmup_factor = float(epoch_no + 1) / float(warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr * warmup_factor
        elif warmup_epochs > 0 and epoch_no == warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr
        model.train()
        with tqdm(train_loader, mininterval=1.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            if warmup_epochs <= 0:
                lr_scheduler.step()
            elif epoch_no + 1 >= warmup_epochs:
                lr_scheduler.step(epoch_no + 1)
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            valid_batch_count = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        valid_batch_count = batch_no
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            current_valid_loss = avg_loss_valid / max(valid_batch_count, 1)
            if best_valid_loss > current_valid_loss:
                best_valid_loss = current_valid_loss
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
                    saved_best_model = True
                print(
                    "\n best loss is updated to ",
                    current_valid_loss,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        if valid_loader is None or not saved_best_model:
            torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def _masked_history_stats(target, eval_points, observed_points):
    hist_mask = (observed_points * (1.0 - eval_points)).float()
    denom = hist_mask.sum(dim=1).clamp(min=1.0)
    hist_mean = (target * hist_mask).sum(dim=1) / denom
    L = target.shape[1]
    time_index = torch.arange(L, device=target.device).view(1, L, 1).expand_as(target)
    first_index = torch.where(hist_mask > 0, time_index, torch.full_like(time_index, L)).min(dim=1).values
    last_index = torch.where(hist_mask > 0, time_index, torch.full_like(time_index, -1)).max(dim=1).values
    first_index = first_index.long().clamp(min=0, max=max(L - 1, 0))
    last_index = last_index.long().clamp(min=0, max=max(L - 1, 0))
    first_value = torch.gather(target, 1, first_index.unsqueeze(1)).squeeze(1)
    last_value = torch.gather(target, 1, last_index.unsqueeze(1)).squeeze(1)
    prev_index = (last_index - 1).clamp(min=0)
    prev_value = torch.gather(target, 1, prev_index.unsqueeze(1)).squeeze(1)
    hist_count = hist_mask.sum(dim=1)
    return hist_mask, hist_mean, first_value, last_value, prev_value, first_index, last_index, hist_count


def _seasonal_copy(target, last_index, hist_count, future_rank, period):
    period = int(period)
    step = future_rank.clamp(min=1).long() - 1
    gather_index = last_index.unsqueeze(1) - period + 1 + torch.remainder(step, period)
    gather_index = gather_index.clamp(min=0, max=max(target.shape[1] - 1, 0))
    copied = torch.gather(target, 1, gather_index.long())
    fallback = torch.gather(target, 1, last_index.long().unsqueeze(1)).expand_as(copied)
    valid = (hist_count > period).unsqueeze(1).expand_as(copied)
    return torch.where(valid, copied, fallback)


def build_forecast_candidates(samples, target, eval_points, observed_points):
    sample_mean = samples.mean(dim=1)
    sample_median = samples.median(dim=1).values
    (
        _hist_mask,
        hist_mean,
        first_value,
        last_value,
        prev_value,
        first_index,
        last_index,
        hist_count,
    ) = _masked_history_stats(target, eval_points, observed_points)
    future_rank = eval_points.cumsum(dim=1).clamp(min=0)
    steps = future_rank.clamp(min=1)
    denom = (last_index - first_index).float().clamp(min=1.0)
    slope = (last_value - first_value) / denom
    last = last_value.unsqueeze(1).expand_as(target)
    trend = last_value.unsqueeze(1) + slope.unsqueeze(1) * steps
    hist_mean_full = hist_mean.unsqueeze(1).expand_as(target)
    phi = torch.exp(-steps / max(float(eval_points.shape[1]), 1.0))
    mean_revert = hist_mean.unsqueeze(1) + phi * (last_value - hist_mean).unsqueeze(1)
    event_decay = last_value.unsqueeze(1) + phi * (last_value - prev_value).unsqueeze(1)
    candidates = [
        sample_mean,
        sample_median,
        last,
        trend,
        hist_mean_full,
        mean_revert,
        event_decay,
    ]
    for period in [2, 3, 4, 6, 7, 12, 24]:
        candidates.append(_seasonal_copy(target, last_index, hist_count, future_rank, period))
    return torch.stack(candidates, dim=-1)


def _append_model_side_candidates(model, batch, candidates, target, eval_points, observed_points, include_timestamp=False):
    candidate_names = list(FORECAST_CANDIDATE_NAMES)
    if not include_timestamp:
        return candidates, candidate_names
    if not getattr(model, "timestep_branch", False):
        return candidates, candidate_names
    if not isinstance(batch, dict) or "timesteps" not in batch or not hasattr(model, "timestep_pred"):
        return candidates, candidate_names
    timesteps = batch["timesteps"].to(target.device).float()
    if timesteps.dim() != 3:
        return candidates, candidate_names
    try:
        ts_pred = model.timestep_pred(timesteps.permute(0, 2, 1)).permute(0, 2, 1)
    except RuntimeError:
        return candidates, candidate_names
    if ts_pred.shape != target.shape:
        return candidates, candidate_names
    hist_mask = (observed_points * (1.0 - eval_points)).float()
    denom = hist_mask.sum(dim=1).clamp(min=1.0)
    mean = (target * hist_mask).sum(dim=1) / denom
    var = (((target - mean.unsqueeze(1)) * hist_mask) ** 2).sum(dim=1) / (denom - 1.0).clamp(min=1.0)
    stdev = torch.sqrt(var + 1e-5)
    ts_pred = ts_pred * stdev.unsqueeze(1) + mean.unsqueeze(1)
    candidates = torch.cat([candidates, ts_pred.unsqueeze(-1)], dim=-1)
    candidate_names.append("timestamp_branch")
    return candidates, candidate_names


def _fit_ridge(X, y, ridge_alpha):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return None
    X_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    penalty = np.eye(X_aug.shape[1], dtype=np.float64) * float(max(ridge_alpha, 0.0))
    penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(X_aug.T @ X_aug + penalty, X_aug.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(X_aug.T @ X_aug + penalty) @ X_aug.T @ y


def _apply_forecast_calibrator(candidates, eval_points, calibrator, base_prediction=None):
    if not calibrator or float(calibrator.get("apply_strength", 0.0)) <= 0:
        return base_prediction
    coeffs = calibrator.get("coefficients")
    if coeffs is None:
        return base_prediction
    coeffs = torch.as_tensor(coeffs, device=candidates.device, dtype=candidates.dtype)
    if coeffs.ndim != 2 or coeffs.shape[1] != candidates.shape[-1] + 1:
        return base_prediction
    strength = float(min(max(calibrator.get("apply_strength", 1.0), 0.0), 1.0))
    residual_clip = float(calibrator.get("residual_clip", 0.0) or 0.0)
    base = candidates[..., 0] if base_prediction is None else base_prediction
    calibrated = base.clone()
    future_rank = eval_points.cumsum(dim=1).long() - 1
    max_h = min(coeffs.shape[0], int(future_rank.max().detach().cpu().item()) + 1 if future_rank.numel() > 0 else 0)
    for h in range(max_h):
        mask = (eval_points > 0) & (future_rank == h)
        if not mask.any():
            continue
        pred_h = coeffs[h, 0] + (candidates * coeffs[h, 1:].view(1, 1, 1, -1)).sum(dim=-1)
        if residual_clip > 0:
            pred_h = base + (pred_h - base).clamp(min=-residual_clip, max=residual_clip)
        calibrated = torch.where(mask, pred_h, calibrated)
    return strength * calibrated + (1.0 - strength) * base


def _masked_mse(prediction, target, eval_points):
    denom = eval_points.sum().clamp(min=1.0)
    return ((((prediction - target) * eval_points) ** 2).sum() / denom).item()


def fit_forecast_calibrator(
    model,
    valid_loader,
    nsample=100,
    foldername="",
    guide_w=0,
    ridge_alpha=1e-3,
    min_gain=0.0,
    max_strength=1.0,
    max_batches=0,
    holdout_fraction=0.35,
    residual_clip_quantile=0.95,
    include_timestamp=False,
):
    if valid_loader is None:
        return None
    if foldername:
        model.load_state_dict(torch.load(foldername + "/model.pth"))
    model.eval()
    rows_by_horizon = {}
    target_by_horizon = {}
    candidate_names = list(FORECAST_CANDIDATE_NAMES)
    mse_mean_num = mse_median_num = mse_base_count = 0.0
    with torch.no_grad():
        with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, valid_batch in enumerate(it, start=1):
                output = model.evaluate(valid_batch, nsample, guide_w)
                if len(output) > 5:
                    samples, c_target, eval_points, observed_points, _observed_time = output[:5]
                else:
                    samples, c_target, eval_points, observed_points, _observed_time = output
                samples = samples.permute(0, 1, 3, 2)
                c_target = c_target.permute(0, 2, 1)
                eval_points = eval_points.permute(0, 2, 1).float()
                observed_points = observed_points.permute(0, 2, 1).float()
                candidates = build_forecast_candidates(samples, c_target, eval_points, observed_points)
                candidates, candidate_names = _append_model_side_candidates(
                    model,
                    valid_batch,
                    candidates,
                    c_target,
                    eval_points,
                    observed_points,
                    include_timestamp=include_timestamp,
                )
                sample_mean = candidates[..., 0]
                sample_median = candidates[..., 1]
                mse_mean_num += ((((sample_mean - c_target) * eval_points) ** 2).sum()).item()
                mse_median_num += ((((sample_median - c_target) * eval_points) ** 2).sum()).item()
                mse_base_count += eval_points.sum().item()
                future_rank = eval_points.cumsum(dim=1).long() - 1
                for h in range(int(future_rank.max().detach().cpu().item()) + 1 if future_rank.numel() > 0 else 0):
                    mask = (eval_points > 0) & (future_rank == h)
                    if not mask.any():
                        continue
                    rows_by_horizon.setdefault(h, []).append(candidates[mask].detach().cpu().numpy())
                    target_by_horizon.setdefault(h, []).append(c_target[mask].detach().cpu().numpy())
                if max_batches and batch_no >= int(max_batches):
                    break

    if not rows_by_horizon or mse_base_count <= 0:
        return None
    max_horizon = max(rows_by_horizon) + 1
    coeffs = np.zeros((max_horizon, len(candidate_names) + 1), dtype=np.float64)
    coeffs[:, 1] = 1.0
    holdout_rows = {}
    holdout_targets = {}
    fit_mse_calibrated_num = fit_count = 0.0
    for h in range(max_horizon):
        if h not in rows_by_horizon:
            continue
        X = np.concatenate(rows_by_horizon[h], axis=0)
        y = np.concatenate(target_by_horizon[h], axis=0)
        min_fit = max(8, X.shape[1] + 2)
        if X.shape[0] < min_fit + 4:
            continue
        holdout_size = int(math.ceil(X.shape[0] * min(max(float(holdout_fraction), 0.1), 0.5)))
        holdout_size = min(max(holdout_size, 4), max(X.shape[0] - min_fit, 0))
        if holdout_size <= 0:
            continue
        split = X.shape[0] - holdout_size
        X_fit, y_fit = X[:split], y[:split]
        X_hold, y_hold = X[split:], y[split:]
        coef = _fit_ridge(X_fit, y_fit, ridge_alpha)
        if coef is not None and np.all(np.isfinite(coef)):
            coeffs[h] = coef
            X_fit_aug = np.concatenate([np.ones((X_fit.shape[0], 1), dtype=np.float64), X_fit], axis=1)
            fit_pred = X_fit_aug @ coeffs[h]
            fit_mse_calibrated_num += float(np.square(fit_pred - y_fit).sum())
            fit_count += float(y_fit.shape[0])
        holdout_rows[h] = X_hold
        holdout_targets[h] = y_hold

    valid_mse_mean = mse_mean_num / max(mse_base_count, 1.0)
    valid_mse_median = mse_median_num / max(mse_base_count, 1.0)
    holdout_mean_num = holdout_median_num = holdout_cal_num = holdout_count = 0.0
    residual_pool = []
    for h, X in holdout_rows.items():
        y = holdout_targets[h]
        X_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
        pred = X_aug @ coeffs[h]
        holdout_mean_num += float(np.square(X[:, 0] - y).sum())
        holdout_median_num += float(np.square(X[:, 1] - y).sum())
        holdout_cal_num += float(np.square(pred - y).sum())
        holdout_count += float(y.shape[0])
    holdout_mse_mean = holdout_mean_num / max(holdout_count, 1.0)
    holdout_mse_median = holdout_median_num / max(holdout_count, 1.0)
    valid_mse_calibrated = holdout_cal_num / max(holdout_count, 1.0)
    fit_mse_calibrated = fit_mse_calibrated_num / max(fit_count, 1.0)
    if holdout_mse_mean <= holdout_mse_median:
        base_mse = holdout_mse_mean
        base_estimator = "mean"
    else:
        base_mse = holdout_mse_median
        base_estimator = "median"
    for h, X in holdout_rows.items():
        y = holdout_targets[h]
        base = X[:, 0] if base_estimator == "mean" else X[:, 1]
        residual_pool.append(np.abs(y - base))
    if residual_pool:
        residual_clip = float(np.quantile(np.concatenate(residual_pool), min(max(float(residual_clip_quantile), 0.5), 1.0)))
    else:
        residual_clip = 0.0
    gain = max((base_mse - valid_mse_calibrated) / max(base_mse, 1e-8), 0.0)
    if gain <= float(max(min_gain, 0.0)):
        strength = 0.0
    else:
        strength = float(min(max(max_strength, 0.0), 1.0, gain))
    return {
        "enabled": True,
        "guarded": True,
        "candidate_names": list(candidate_names),
        "ridge_alpha": float(ridge_alpha),
        "holdout_fraction": float(holdout_fraction),
        "valid_mse_mean": float(valid_mse_mean),
        "valid_mse_median": float(valid_mse_median),
        "valid_mse_calibrated": float(valid_mse_calibrated),
        "fit_mse_calibrated": float(fit_mse_calibrated),
        "holdout_mse_mean": float(holdout_mse_mean),
        "holdout_mse_median": float(holdout_mse_median),
        "valid_gain": float(gain),
        "base_estimator": base_estimator,
        "apply_strength": strength,
        "residual_clip": float(residual_clip),
        "residual_clip_quantile": float(residual_clip_quantile),
        "include_timestamp": bool(include_timestamp),
        "coefficients": coeffs.tolist(),
    }


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", window_lens=[1, 1], guide_w=0, save_attn=False, save_token=False, save_trend_prior=False, point_estimator="mean", forecast_calibrator=None):
    model.load_state_dict(torch.load(foldername + "/model.pth"))
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        mse_mean_total = 0
        mse_median_total = 0
        mse_calibrated_total = 0
        nmse_total = 0
        nmae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        all_tt_attns = []
        all_tf_attns = []
        all_tokens = []
        all_trend_priors = []
        all_text_marks = []
        with tqdm(test_loader, mininterval=1.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample, guide_w)

                if save_trend_prior and isinstance(test_batch, dict) and "trend_prior" in test_batch:
                    all_trend_priors.append(test_batch["trend_prior"].detach().cpu().numpy())
                    if "text_mark" in test_batch:
                        all_text_marks.append(test_batch["text_mark"].detach().cpu().numpy())

                if save_attn:
                    if save_token:
                        samples, c_target, eval_points, observed_points, observed_time, attns, tokens = output
                    else:
                        samples, c_target, eval_points, observed_points, observed_time, attns = output
                else:
                    samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                candidates = build_forecast_candidates(samples, c_target, eval_points.float(), observed_points.float())
                candidates, _candidate_names = _append_model_side_candidates(
                    model,
                    test_batch,
                    candidates,
                    c_target,
                    eval_points.float(),
                    observed_points.float(),
                    include_timestamp=bool(
                        forecast_calibrator
                        and forecast_calibrator.get("include_timestamp", False)
                        and "timestamp_branch" in forecast_calibrator.get("candidate_names", [])
                    ),
                )
                samples_mean = candidates[..., 0]
                samples_median = candidates[..., 1]
                estimator_name = str(point_estimator or "mean").lower()
                if estimator_name in ("auto", "valid_auto") and forecast_calibrator is not None:
                    estimator_name = str(forecast_calibrator.get("base_estimator", "mean")).lower()
                if estimator_name == "median":
                    point_prediction = samples_median
                else:
                    estimator_name = "mean"
                    point_prediction = samples_mean
                calibrated_prediction = _apply_forecast_calibrator(
                    candidates,
                    eval_points.float(),
                    forecast_calibrator,
                    base_prediction=point_prediction,
                )
                if calibrated_prediction is not None and forecast_calibrator and float(forecast_calibrator.get("apply_strength", 0.0)) > 0:
                    point_prediction = calibrated_prediction
                    estimator_name = "calibrated"
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)
                if save_attn:
                    f = lambda x: x.detach().mean(dim=1).unsqueeze(1)
                    attns = [(f(attn1), f(attn2)) for attn1, attn2 in attns] 
                    tt_attns, tf_attns = zip(*attns)
                    tt_attns = torch.cat(tt_attns, 1)
                    tf_attns = torch.cat(tf_attns, 1)
                    tt_attns = tt_attns.chunk(2, dim=0)[0]
                    tf_attns = tf_attns.chunk(2, dim=0)[0]
                    all_tt_attns.append(tt_attns) 
                    all_tf_attns.append(tf_attns) 
                if save_token:
                    all_tokens.extend(tokens)

                mse_current = (
                    ((point_prediction - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((point_prediction - c_target) * eval_points)
                ) * scaler
                nmse_current = (
                    ((point_prediction - c_target) * eval_points) ** 2
                )
                nmae_current = (
                    torch.abs((point_prediction - c_target) * eval_points)
                )
                mse_mean_current = ((samples_mean - c_target) * eval_points) ** 2
                mse_median_current = ((samples_median - c_target) * eval_points) ** 2
                if calibrated_prediction is None:
                    calibrated_prediction = point_prediction
                mse_calibrated_current = ((calibrated_prediction - c_target) * eval_points) ** 2

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                mse_mean_total += mse_mean_current.sum().item()
                mse_median_total += mse_median_current.sum().item()
                mse_calibrated_total += mse_calibrated_current.sum().item()
                nmse_total += nmse_current.sum().item()
                nmae_total += nmae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "nmse_total": nmse_total / evalpoints_total,
                        "nmae_total": nmae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            all_target = torch.cat(all_target, dim=0)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)
            all_observed_point = torch.cat(all_observed_point, dim=0)
            all_observed_time = torch.cat(all_observed_time, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)
            # if save_attn:
            #     all_tt_attns = torch.cat(all_tt_attns, dim=0)
            #     all_tf_attns = torch.cat(all_tf_attns, dim=0)


            # np.save(foldername + "/generated_nsample" + str(nsample) + "_guide" + str(guide_w) + ".npy", all_generated_samples.cpu().numpy())
            # np.save(foldername + "/target_" + str(nsample) + "_guide" + str(guide_w) + ".npy", all_target.cpu().numpy())
            # if save_attn:
            #     np.save(foldername + "/all_tt_attns" + ".npy", all_tt_attns.cpu().numpy())
            #     np.save(foldername + "/all_tf_attns" + ".npy", all_tf_attns.cpu().numpy())
            # if save_token:
            #     np.save(foldername + "/tokens" + ".npy", np.asarray(all_tokens))
            if save_trend_prior and all_trend_priors:
                trend_prior_arr = np.concatenate(all_trend_priors, axis=0)
                np.save(foldername + "trend_priors.npy", trend_prior_arr)
                if all_text_marks:
                    text_mark_arr = np.concatenate(all_text_marks, axis=0)
                    np.save(foldername + "trend_text_marks.npy", text_mark_arr)

            crps = calc_quantile_CRPS(
                all_target,
                all_generated_samples,
                all_evalpoint,
                mean_scaler,
                scaler,
            )

            results = {
                "guide_w": guide_w,
                "CRPS": crps,
                "MSE": nmse_total / evalpoints_total,
                "MAE": nmae_total / evalpoints_total,
                "MSE_mean": mse_mean_total / evalpoints_total,
                "MSE_median": mse_median_total / evalpoints_total,
                "MSE_calibrated": mse_calibrated_total / evalpoints_total,
                "point_estimator": estimator_name,
            }
            if forecast_calibrator is not None:
                results["forecast_calibrator"] = {
                    key: value for key, value in forecast_calibrator.items() if key != "coefficients"
                }
            with open(foldername + "metrics.json", "w") as f:
                json.dump(results, f, indent=4)
            print("CRPS:", crps)
            print("MSE:", nmse_total / evalpoints_total)
            print("MAE:", nmae_total / evalpoints_total)
    return results
