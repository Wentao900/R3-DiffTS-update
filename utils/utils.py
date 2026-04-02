import numpy as np
import torch
from torch.optim import Adam, AdamW
from tqdm import tqdm
import os
import json


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

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.current_epoch = epoch_no
        model.total_epochs = max(int(config["epochs"]), 1)
        model.train()
        with tqdm(train_loader, mininterval=1.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
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

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
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


def _get_rerank_config(model):
    cfg = getattr(model, "config", None)
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    rerank_enabled = bool(model_cfg.get("rerank_samples", False))
    rerank_topk = int(model_cfg.get("rerank_topk", 1))
    rerank_boundary_weight = float(model_cfg.get("rerank_boundary_weight", 1.0))
    rerank_volatility_weight = float(model_cfg.get("rerank_volatility_weight", 0.25))
    rerank_use_median = bool(model_cfg.get("rerank_use_median", True))
    return {
        "enabled": rerank_enabled,
        "topk": rerank_topk,
        "boundary_weight": rerank_boundary_weight,
        "volatility_weight": rerank_volatility_weight,
        "use_median": rerank_use_median,
    }


def rerank_samples(samples, observed_points, model):
    rerank_cfg = _get_rerank_config(model)
    if (not rerank_cfg["enabled"]) or samples.shape[1] <= 1:
        return samples.median(dim=1)

    topk = max(1, min(int(rerank_cfg["topk"]), samples.shape[1]))
    lookback_len = int(getattr(model, "lookback_len", 0))
    pred_len = int(getattr(model, "pred_len", 0))
    if lookback_len <= 0 or pred_len <= 0 or lookback_len >= samples.shape[2]:
        return samples.median(dim=1)

    history_last = observed_points[:, lookback_len - 1, :].unsqueeze(1)
    forecast_first = samples[:, :, lookback_len, :]
    boundary_jump = (forecast_first - history_last).abs().mean(dim=2)

    future = samples[:, :, lookback_len:, :]
    if future.shape[2] > 1:
        future_diffs = future[:, :, 1:, :] - future[:, :, :-1, :]
        future_volatility = future_diffs.abs().mean(dim=(2, 3))
    else:
        future_volatility = torch.zeros_like(boundary_jump)

    score = (
        rerank_cfg["boundary_weight"] * boundary_jump
        + rerank_cfg["volatility_weight"] * future_volatility
    )
    topk_idx = torch.topk(score, k=topk, dim=1, largest=False).indices
    topk_idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, samples.shape[2], samples.shape[3])
    selected_samples = torch.gather(samples, 1, topk_idx)

    if rerank_cfg["use_median"]:
        return selected_samples.median(dim=1).values
    return selected_samples.mean(dim=1)

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", window_lens=[1, 1], guide_w=0, save_attn=False, save_token=False, save_trend_prior=False):
    model.load_state_dict(torch.load(foldername + "/model.pth"))
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
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

                aggregated_samples = rerank_samples(samples, observed_points, model)
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
                    ((aggregated_samples - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((aggregated_samples - c_target) * eval_points) 
                ) * scaler
                nmse_current = (
                    ((aggregated_samples - c_target) * eval_points) ** 2
                )
                nmae_current = (
                    torch.abs((aggregated_samples - c_target) * eval_points) 
                )

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
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

            results = {
                "guide_w": guide_w,
                "MSE": nmse_total / evalpoints_total,
                "MAE": nmae_total / evalpoints_total,
            }
            with open(foldername + "config_results.json", "a") as f:
                json.dump(results, f, indent=4)
            print("MSE:", nmse_total / evalpoints_total)
            print("MAE:", nmae_total / evalpoints_total)
    return nmse_total / evalpoints_total
