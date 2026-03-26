import csv
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
    use_amp=True,
):
    optimizer = Adam(model.parameters(), lr=float(config["lr"]), weight_decay=1e-6)

    # FP16 mixed precision setup
    device_is_cuda = next(model.parameters()).is_cuda if len(list(model.parameters())) > 0 else False
    amp_enabled = use_amp and device_is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if amp_enabled:
        print("[AMP] FP16 mixed precision training ENABLED")
    else:
        print("[AMP] Mixed precision training disabled (CPU or use_amp=False)")
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
        model.train()
        with tqdm(train_loader, mininterval=1.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss = model(train_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                avg_loss += loss.item()
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
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
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


def _clone_counterfactual_batch(batch):
    cloned = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        elif isinstance(value, tuple):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def _is_nonempty_text(text):
    if text is None:
        return False
    value = str(text).strip()
    return len(value) > 0 and value.upper() != "NA"


def _build_counterfactual_batch(batch, mode):
    variant = _clone_counterfactual_batch(batch)
    batch_size = None
    if "text_mark" in batch and torch.is_tensor(batch["text_mark"]):
        batch_size = int(batch["text_mark"].shape[0])
    elif "texts" in batch and isinstance(batch["texts"], (list, tuple)):
        batch_size = len(batch["texts"])
    else:
        raise ValueError("Unable to infer batch size for counterfactual evaluation.")

    if mode == "full_text":
        return variant

    if mode == "text_off":
        variant["texts"] = ["NA"] * batch_size
        variant["retrieved_text"] = [""] * batch_size
        variant["cot_text"] = [""] * batch_size
        if "text_mark" in variant and torch.is_tensor(variant["text_mark"]):
            variant["text_mark"] = torch.zeros_like(variant["text_mark"])
        if "trend_prior" in variant and torch.is_tensor(variant["trend_prior"]):
            variant["trend_prior"] = torch.zeros_like(variant["trend_prior"])
        if "text_score" in variant and torch.is_tensor(variant["text_score"]):
            variant["text_score"] = torch.zeros_like(variant["text_score"])
        return variant

    if mode == "raw_only":
        raw_texts = variant.get("raw_text", variant.get("texts"))
        if isinstance(raw_texts, tuple):
            raw_texts = list(raw_texts)
        elif not isinstance(raw_texts, list):
            raw_texts = [raw_texts] * batch_size
        raw_texts = [str(item) if item is not None else "NA" for item in raw_texts]
        variant["texts"] = raw_texts
        variant["retrieved_text"] = [""] * batch_size
        variant["cot_text"] = [""] * batch_size
        marks = [1 if _is_nonempty_text(item) else 0 for item in raw_texts]
        if "text_mark" in variant and torch.is_tensor(variant["text_mark"]):
            variant["text_mark"] = torch.as_tensor(marks, dtype=variant["text_mark"].dtype, device=variant["text_mark"].device)
        if "trend_prior" in variant and torch.is_tensor(variant["trend_prior"]):
            variant["trend_prior"] = torch.zeros_like(variant["trend_prior"])
        return variant

    raise ValueError(f"Unsupported counterfactual mode: {mode}")


def _unpack_eval_output(output, save_attn=False, save_token=False):
    if save_attn:
        if save_token:
            samples, c_target, eval_points, observed_points, observed_time, _, _ = output
        else:
            samples, c_target, eval_points, observed_points, observed_time, _ = output
    else:
        samples, c_target, eval_points, observed_points, observed_time = output
    return samples, c_target, eval_points, observed_points, observed_time


def _token_set(text):
    value = str(text).strip().lower()
    if not value or value == "na":
        return set()
    return set(value.split())


def _safe_ratio(num, den):
    return float(num) / float(den) if den else 0.0


def _compute_batch_sample_metrics(output, scaler):
    samples, c_target, eval_points, _, _ = _unpack_eval_output(output, save_attn=False, save_token=False)
    samples = samples.permute(0, 1, 3, 2)
    c_target = c_target.permute(0, 2, 1)
    eval_points = eval_points.permute(0, 2, 1)
    samples_median = samples.median(dim=1).values
    sq = (((samples_median - c_target) * eval_points) ** 2) * (scaler ** 2)
    ab = torch.abs((samples_median - c_target) * eval_points) * scaler
    denom = eval_points.sum(dim=(1, 2)).detach().cpu().numpy()
    mse_num = sq.sum(dim=(1, 2)).detach().cpu().numpy()
    mae_num = ab.sum(dim=(1, 2)).detach().cpu().numpy()
    safe = np.maximum(denom, 1e-8)
    return {
        "mse": mse_num / safe,
        "mae": mae_num / safe,
        "eval_count": denom,
    }


def _safe_group_value(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        if np.isnan(value):
            return "NA"
        if value.is_integer():
            return str(int(value))
    return str(value)


def _summarize_counterfactual_groups(rows, modes, group_key):
    grouped = {}
    for row in rows:
        group_value = _safe_group_value(row.get(group_key))
        stats = grouped.setdefault(group_value, {"sample_count": 0, "modes": {}})
        stats["sample_count"] += 1
        for mode in modes:
            mode_stats = stats["modes"].setdefault(
                mode,
                {
                    "mse_sum": 0.0,
                    "mae_sum": 0.0,
                    "mse_count": 0,
                    "mae_count": 0,
                    "delta_mse_sum": 0.0,
                    "delta_mse_pos": 0,
                    "delta_mse_neg": 0,
                    "delta_mae_sum": 0.0,
                },
            )
            mse_key = f"mse_{mode}"
            mae_key = f"mae_{mode}"
            if mse_key in row:
                mse_value = float(row[mse_key])
                mode_stats["mse_sum"] += mse_value
                mode_stats["mse_count"] += 1
            if mae_key in row:
                mae_value = float(row[mae_key])
                mode_stats["mae_sum"] += mae_value
                mode_stats["mae_count"] += 1
            if mode == "text_off":
                continue
            delta_mse = float(row[f"delta_mse_{mode}"])
            delta_mae = float(row[f"delta_mae_{mode}"])
            mode_stats["delta_mse_sum"] += delta_mse
            mode_stats["delta_mae_sum"] += delta_mae
            if delta_mse > 0.0:
                mode_stats["delta_mse_pos"] += 1
            elif delta_mse < 0.0:
                mode_stats["delta_mse_neg"] += 1

    summary = {}
    for group_value, group_stats in grouped.items():
        summary[group_value] = {"sample_count": int(group_stats["sample_count"]), "modes": {}}
        for mode, mode_stats in group_stats["modes"].items():
            mode_summary = {}
            if mode_stats["mse_count"] > 0:
                mode_summary["MSE"] = float(mode_stats["mse_sum"] / mode_stats["mse_count"])
            if mode_stats["mae_count"] > 0:
                mode_summary["MAE"] = float(mode_stats["mae_sum"] / mode_stats["mae_count"])
            if mode != "text_off" and group_stats["sample_count"] > 0:
                count = float(group_stats["sample_count"])
                mode_summary["delta_vs_text_off"] = {
                    "mean_delta_mse": float(mode_stats["delta_mse_sum"] / count),
                    "mean_delta_mae": float(mode_stats["delta_mae_sum"] / count),
                    "positive_gain_rate": float(mode_stats["delta_mse_pos"] / count),
                    "negative_gain_rate": float(mode_stats["delta_mse_neg"] / count),
                }
            summary[group_value]["modes"][mode] = mode_summary
    return summary


def evaluate_counterfactual(
    model,
    test_loader,
    nsample=100,
    scaler=1,
    mean_scaler=0,
    foldername="",
    guide_w=0,
    model_folder=None,
    split="test",
    modes=None,
):
    del mean_scaler
    load_folder = model_folder or foldername
    if load_folder:
        model_path = os.path.join(load_folder, "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=model.device))

    requested_modes = list(modes) if modes else ["text_off", "raw_only", "full_text"]
    ordered_modes = []
    for mode in ["text_off", "raw_only", "full_text"]:
        if mode in requested_modes and mode not in ordered_modes:
            ordered_modes.append(mode)
    for mode in requested_modes:
        if mode not in ordered_modes:
            ordered_modes.append(mode)
    if "text_off" not in ordered_modes:
        ordered_modes.insert(0, "text_off")

    with torch.no_grad():
        model.eval()
        summary = {mode: {"mse_sum": 0.0, "mae_sum": 0.0, "eval_sum": 0.0, "sample_count": 0} for mode in ordered_modes}
        rows = []
        sample_offset = 0
        with tqdm(test_loader, mininterval=1.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                batch_metrics = {}
                batch_size = None
                for mode in ordered_modes:
                    variant_batch = _build_counterfactual_batch(test_batch, mode)
                    output = model.evaluate(variant_batch, nsample, guide_w)
                    metrics = _compute_batch_sample_metrics(output, scaler)
                    batch_metrics[mode] = metrics
                    summary[mode]["mse_sum"] += float(np.sum(metrics["mse"] * metrics["eval_count"]))
                    summary[mode]["mae_sum"] += float(np.sum(metrics["mae"] * metrics["eval_count"]))
                    summary[mode]["eval_sum"] += float(np.sum(metrics["eval_count"]))
                    summary[mode]["sample_count"] += int(len(metrics["mse"]))
                    if batch_size is None:
                        batch_size = len(metrics["mse"])

                base_mse = batch_metrics["text_off"]["mse"]
                base_mae = batch_metrics["text_off"]["mae"]
                text_marks = test_batch.get("text_mark")
                if torch.is_tensor(text_marks):
                    text_marks = text_marks.detach().cpu().numpy()
                else:
                    text_marks = np.zeros(batch_size, dtype=np.int64)
                text_window_len = test_batch.get("text_window_len")
                if torch.is_tensor(text_window_len):
                    text_window_len = text_window_len.detach().cpu().numpy()
                else:
                    text_window_len = np.zeros(batch_size, dtype=np.int64)
                scale_code = test_batch.get("scale_code")
                if torch.is_tensor(scale_code):
                    scale_code = scale_code.detach().cpu().numpy()
                else:
                    scale_code = np.zeros(batch_size, dtype=np.int64)
                trend_prior = test_batch.get("trend_prior")
                if torch.is_tensor(trend_prior):
                    trend_prior = trend_prior.detach().cpu().numpy()
                else:
                    trend_prior = np.zeros((batch_size, 3), dtype=np.float32)
                scale_pref = test_batch.get("scale_pref")
                if torch.is_tensor(scale_pref):
                    scale_pref = scale_pref.detach().cpu().numpy()
                else:
                    scale_pref = np.zeros(batch_size, dtype=np.float32)
                signed_slope = test_batch.get("signed_slope")
                if torch.is_tensor(signed_slope):
                    signed_slope = signed_slope.detach().cpu().numpy()
                else:
                    signed_slope = np.zeros(batch_size, dtype=np.float32)
                abs_slope = test_batch.get("abs_slope")
                if torch.is_tensor(abs_slope):
                    abs_slope = abs_slope.detach().cpu().numpy()
                else:
                    abs_slope = np.zeros(batch_size, dtype=np.float32)
                history_std = test_batch.get("history_std")
                if torch.is_tensor(history_std):
                    history_std = history_std.detach().cpu().numpy()
                else:
                    history_std = np.zeros(batch_size, dtype=np.float32)
                history_mean_abs = test_batch.get("history_mean_abs")
                if torch.is_tensor(history_mean_abs):
                    history_mean_abs = history_mean_abs.detach().cpu().numpy()
                else:
                    history_mean_abs = np.zeros(batch_size, dtype=np.float32)
                history_total_shift = test_batch.get("history_total_shift")
                if torch.is_tensor(history_total_shift):
                    history_total_shift = history_total_shift.detach().cpu().numpy()
                else:
                    history_total_shift = np.zeros(batch_size, dtype=np.float32)
                history_accel = test_batch.get("history_accel")
                if torch.is_tensor(history_accel):
                    history_accel = history_accel.detach().cpu().numpy()
                else:
                    history_accel = np.zeros(batch_size, dtype=np.float32)
                history_smoothness = test_batch.get("history_smoothness")
                if torch.is_tensor(history_smoothness):
                    history_smoothness = history_smoothness.detach().cpu().numpy()
                else:
                    history_smoothness = np.zeros(batch_size, dtype=np.float32)
                history_trend_score = test_batch.get("history_trend_score")
                if torch.is_tensor(history_trend_score):
                    history_trend_score = history_trend_score.detach().cpu().numpy()
                else:
                    history_trend_score = np.zeros(batch_size, dtype=np.float32)
                history_volatility_score = test_batch.get("history_volatility_score")
                if torch.is_tensor(history_volatility_score):
                    history_volatility_score = history_volatility_score.detach().cpu().numpy()
                else:
                    history_volatility_score = np.zeros(batch_size, dtype=np.float32)
                history_last_value = test_batch.get("history_last_value")
                if torch.is_tensor(history_last_value):
                    history_last_value = history_last_value.detach().cpu().numpy()
                else:
                    history_last_value = np.zeros(batch_size, dtype=np.float32)
                raw_texts = test_batch.get("raw_text", ["NA"] * batch_size)
                full_texts = test_batch.get("texts", ["NA"] * batch_size)
                retrieved_texts = test_batch.get("retrieved_text", [""] * batch_size)
                cot_texts = test_batch.get("cot_text", [""] * batch_size)
                if isinstance(raw_texts, tuple):
                    raw_texts = list(raw_texts)
                if isinstance(full_texts, tuple):
                    full_texts = list(full_texts)
                if isinstance(retrieved_texts, tuple):
                    retrieved_texts = list(retrieved_texts)
                if isinstance(cot_texts, tuple):
                    cot_texts = list(cot_texts)

                for idx in range(batch_size):
                    raw_token_set = _token_set(raw_texts[idx])
                    full_token_set = _token_set(full_texts[idx])
                    retrieved_token_set = _token_set(retrieved_texts[idx])
                    cot_token_set = _token_set(cot_texts[idx])
                    overlap_raw_retrieved = len(raw_token_set & retrieved_token_set)
                    overlap_full_retrieved = len(full_token_set & retrieved_token_set)
                    row = {
                        "batch_no": batch_no,
                        "sample_idx": sample_offset + idx,
                        "text_mark": int(text_marks[idx]),
                        "text_window_len": int(text_window_len[idx]) if len(text_window_len) > idx else 0,
                        "scale_code": int(scale_code[idx]) if len(scale_code) > idx else 0,
                        "scale_pref": float(scale_pref[idx]) if len(scale_pref) > idx else 0.0,
                        "signed_slope": float(signed_slope[idx]) if len(signed_slope) > idx else 0.0,
                        "abs_slope": float(abs_slope[idx]) if len(abs_slope) > idx else 0.0,
                        "history_std": float(history_std[idx]) if len(history_std) > idx else 0.0,
                        "history_mean_abs": float(history_mean_abs[idx]) if len(history_mean_abs) > idx else 0.0,
                        "history_total_shift": float(history_total_shift[idx]) if len(history_total_shift) > idx else 0.0,
                        "history_accel": float(history_accel[idx]) if len(history_accel) > idx else 0.0,
                        "history_smoothness": float(history_smoothness[idx]) if len(history_smoothness) > idx else 0.0,
                        "history_trend_score": float(history_trend_score[idx]) if len(history_trend_score) > idx else 0.0,
                        "history_volatility_score": float(history_volatility_score[idx]) if len(history_volatility_score) > idx else 0.0,
                        "history_last_value": float(history_last_value[idx]) if len(history_last_value) > idx else 0.0,
                        "raw_text_len": len(str(raw_texts[idx]).split()),
                        "full_text_len": len(str(full_texts[idx]).split()),
                        "retrieved_text_len": len(str(retrieved_texts[idx]).split()),
                        "cot_text_len": len(str(cot_texts[idx]).split()),
                        "extra_text_len": max(len(str(full_texts[idx]).split()) - len(str(raw_texts[idx]).split()), 0),
                        "retrieval_to_raw_len_ratio": _safe_ratio(len(str(retrieved_texts[idx]).split()), len(str(raw_texts[idx]).split())),
                        "cot_to_full_len_ratio": _safe_ratio(len(str(cot_texts[idx]).split()), len(str(full_texts[idx]).split())),
                        "raw_retrieved_overlap": float(overlap_raw_retrieved),
                        "full_retrieved_overlap": float(overlap_full_retrieved),
                        "raw_retrieved_jaccard": _safe_ratio(overlap_raw_retrieved, len(raw_token_set | retrieved_token_set)),
                        "full_retrieved_jaccard": _safe_ratio(overlap_full_retrieved, len(full_token_set | retrieved_token_set)),
                        "retrieved_unique_ratio": _safe_ratio(len(retrieved_token_set), len(str(retrieved_texts[idx]).split())),
                        "full_unique_ratio": _safe_ratio(len(full_token_set), len(str(full_texts[idx]).split())),
                        "cot_unique_ratio": _safe_ratio(len(cot_token_set), len(str(cot_texts[idx]).split())),
                        "trend_direction": float(trend_prior[idx][0]) if len(trend_prior) > idx else 0.0,
                        "trend_strength": float(trend_prior[idx][1]) if len(trend_prior) > idx else 0.0,
                        "trend_volatility": float(trend_prior[idx][2]) if len(trend_prior) > idx else 0.0,
                        "guide_w": float(guide_w),
                        "mse_text_off": float(base_mse[idx]),
                        "mae_text_off": float(base_mae[idx]),
                    }
                    for mode in ordered_modes:
                        if mode == "text_off":
                            continue
                        row[f"mse_{mode}"] = float(batch_metrics[mode]["mse"][idx])
                        row[f"mae_{mode}"] = float(batch_metrics[mode]["mae"][idx])
                        row[f"delta_mse_{mode}"] = float(base_mse[idx] - batch_metrics[mode]["mse"][idx])
                        row[f"delta_mae_{mode}"] = float(base_mae[idx] - batch_metrics[mode]["mae"][idx])
                    rows.append(row)
                sample_offset += batch_size

                display = {"batch_no": batch_no}
                for mode in ordered_modes:
                    denom = max(summary[mode]["eval_sum"], 1e-8)
                    display[f"mse_{mode}"] = summary[mode]["mse_sum"] / denom
                it.set_postfix(ordered_dict=display, refresh=True)

    results = {
        "split": split,
        "guide_w": guide_w,
        "modes": ordered_modes,
        "summary": {},
        "grouped_summary": {},
    }
    for mode in ordered_modes:
        denom = max(summary[mode]["eval_sum"], 1e-8)
        results["summary"][mode] = {
            "MSE": float(summary[mode]["mse_sum"] / denom),
            "MAE": float(summary[mode]["mae_sum"] / denom),
            "sample_count": int(summary[mode]["sample_count"]),
        }
    baseline_rows = rows
    for mode in ordered_modes:
        if mode == "text_off":
            continue
        delta_mse = [row[f"delta_mse_{mode}"] for row in baseline_rows]
        delta_mae = [row[f"delta_mae_{mode}"] for row in baseline_rows]
        results["summary"][mode]["delta_vs_text_off"] = {
            "mean_delta_mse": float(np.mean(delta_mse)) if delta_mse else 0.0,
            "mean_delta_mae": float(np.mean(delta_mae)) if delta_mae else 0.0,
            "positive_gain_rate": float(np.mean(np.asarray(delta_mse) > 0.0)) if delta_mse else 0.0,
            "negative_gain_rate": float(np.mean(np.asarray(delta_mse) < 0.0)) if delta_mse else 0.0,
        }

    if rows:
        for group_key in ["text_mark", "text_window_len", "scale_code"]:
            results["grouped_summary"][group_key] = _summarize_counterfactual_groups(rows, ordered_modes, group_key)

    if foldername:
        metrics_prefix = "eval" if split == "test" else split
        guide_tag = str(guide_w).replace("-", "m").replace(".", "p")
        json_path = os.path.join(foldername, f"{metrics_prefix}_counterfactual_guide_{guide_tag}.json")
        csv_path = os.path.join(foldername, f"{metrics_prefix}_counterfactual_samples_guide_{guide_tag}.csv")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    print("Counterfactual summary:", json.dumps(results, ensure_ascii=True))
    return results

def evaluate(
    model,
    test_loader,
    nsample=100,
    scaler=1,
    mean_scaler=0,
    foldername="",
    window_lens=[1, 1],
    guide_w=0,
    save_attn=False,
    save_token=False,
    save_trend_prior=False,
    model_folder=None,
    split="test",
    append_to_config_results=True,
):
    load_folder = model_folder or foldername
    if load_folder:
        model_path = os.path.join(load_folder, "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=model.device))
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
        pred_len = int(window_lens[1]) if len(window_lens) > 1 else 0
        horizon_mse_total = np.zeros(pred_len, dtype=np.float64)
        horizon_mae_total = np.zeros(pred_len, dtype=np.float64)
        horizon_evalpoints_total = np.zeros(pred_len, dtype=np.float64)
        band_infos = model.get_multi_res_band_info() if hasattr(model, "get_multi_res_band_info") else []
        band_mse_total = {label: 0.0 for label, _ in band_infos}
        band_mae_total = {label: 0.0 for label, _ in band_infos}
        band_evalpoints_total = {label: 0.0 for label, _ in band_infos}
        router_weight_sum = None
        router_argmax_total = None
        router_entropy_sum = 0.0
        router_sample_total = 0
        router_target_hits = 0
        router_target_total = 0
        router_text_window_sum = 0.0
        router_text_window_count = 0
        guide_weight_sum = 0.0
        guide_weight_sq_sum = 0.0
        guide_weight_count = 0
        guide_ratio_sum = 0.0
        guide_ratio_count = 0
        scale_score_sum = 0.0
        scale_score_count = 0
        with tqdm(test_loader, mininterval=1.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample, guide_w)
                if hasattr(model, "get_scale_router_diagnostics"):
                    router_diag = model.get_scale_router_diagnostics(test_batch, guide_w=guide_w)
                else:
                    router_diag = None

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

                samples_median = samples.median(dim=1)
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
                if router_diag is not None:
                    weights = router_diag["weights"].double()
                    argmax = router_diag["argmax"].long()
                    if router_weight_sum is None:
                        router_weight_sum = np.zeros(weights.shape[1], dtype=np.float64)
                        router_argmax_total = np.zeros(weights.shape[1], dtype=np.float64)
                    router_weight_sum += weights.sum(dim=0).cpu().numpy()
                    router_argmax_total += np.bincount(argmax.cpu().numpy(), minlength=weights.shape[1]).astype(np.float64)
                    router_entropy_sum += router_diag["entropy"].double().sum().item()
                    router_sample_total += weights.shape[0]
                    if "target_index" in router_diag:
                        target_index = router_diag["target_index"].long()
                        router_target_hits += (argmax == target_index).sum().item()
                        router_target_total += target_index.numel()
                    if "text_window_len" in router_diag:
                        router_text_window_sum += router_diag["text_window_len"].double().sum().item()
                        router_text_window_count += router_diag["text_window_len"].numel()
                    if "sample_guide_w" in router_diag:
                        sample_guide = router_diag["sample_guide_w"].double()
                        guide_weight_sum += sample_guide.sum().item()
                        guide_weight_sq_sum += (sample_guide ** 2).sum().item()
                        guide_weight_count += sample_guide.numel()
                    if "guide_ratio" in router_diag:
                        guide_ratio = router_diag["guide_ratio"].double()
                        guide_ratio_sum += guide_ratio.sum().item()
                        guide_ratio_count += guide_ratio.numel()
                    if "scale_score" in router_diag:
                        scale_score = router_diag["scale_score"].double()
                        scale_score_sum += scale_score.sum().item()
                        scale_score_count += scale_score.numel()

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler
                nmse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                )
                nmae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                )

                if pred_len > 0:
                    pred_slice = slice(window_lens[0], window_lens[0] + pred_len)
                    pred_sq = nmse_current[:, pred_slice, :].sum(dim=2)
                    pred_abs = nmae_current[:, pred_slice, :].sum(dim=2)
                    pred_eval = eval_points[:, pred_slice, :].sum(dim=2)
                    horizon_mse_total += pred_sq.sum(dim=0).detach().cpu().numpy()
                    horizon_mae_total += pred_abs.sum(dim=0).detach().cpu().numpy()
                    horizon_evalpoints_total += pred_eval.sum(dim=0).detach().cpu().numpy()
                    for label, (start, end) in band_infos:
                        band_mse_total[label] += pred_sq[:, start:end].sum().item()
                        band_mae_total[label] += pred_abs[:, start:end].sum().item()
                        band_evalpoints_total[label] += pred_eval[:, start:end].sum().item()

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

            if not all_target:
                raise RuntimeError(
                    f"{split} loader produced no batches. "
                    "Check dataset size, batch_size, num_workers, and drop_last settings."
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
            if save_trend_prior and all_trend_priors and foldername:
                trend_prior_arr = np.concatenate(all_trend_priors, axis=0)
                np.save(os.path.join(foldername, "trend_priors.npy"), trend_prior_arr)
                if all_text_marks:
                    text_mark_arr = np.concatenate(all_text_marks, axis=0)
                    np.save(os.path.join(foldername, "trend_text_marks.npy"), text_mark_arr)

            horizon_metrics = {}
            for idx in range(pred_len):
                denom = horizon_evalpoints_total[idx]
                if denom <= 0:
                    continue
                horizon_metrics[f"h{idx + 1}"] = {
                    "MSE": float(horizon_mse_total[idx] / denom),
                    "MAE": float(horizon_mae_total[idx] / denom),
                }

            band_metrics = {}
            for label, _ in band_infos:
                denom = band_evalpoints_total.get(label, 0.0)
                if denom <= 0:
                    continue
                band_metrics[label] = {
                    "MSE": float(band_mse_total[label] / denom),
                    "MAE": float(band_mae_total[label] / denom),
                }

            router_metrics = {}
            if router_weight_sum is not None and router_sample_total > 0:
                router_labels = [label for label, _ in band_infos]
                if not router_labels:
                    router_labels = [f"band_{idx}" for idx in range(len(router_weight_sum))]
                router_metrics = {
                    "mean_weights": {
                        label: float(router_weight_sum[idx] / router_sample_total)
                        for idx, label in enumerate(router_labels)
                    },
                    "argmax_freq": {
                        label: float(router_argmax_total[idx] / router_sample_total)
                        for idx, label in enumerate(router_labels)
                    },
                    "mean_entropy": float(router_entropy_sum / router_sample_total),
                }
                if router_target_total > 0:
                    router_metrics["teacher_alignment"] = float(router_target_hits / router_target_total)
                if router_text_window_count > 0:
                    router_metrics["mean_text_window_len"] = float(router_text_window_sum / router_text_window_count)
                if guide_weight_count > 0:
                    mean_sample_guide = guide_weight_sum / guide_weight_count
                    guide_var = max(guide_weight_sq_sum / guide_weight_count - mean_sample_guide ** 2, 0.0)
                    router_metrics["mean_sample_guide_w"] = float(mean_sample_guide)
                    router_metrics["std_sample_guide_w"] = float(np.sqrt(guide_var))
                if guide_ratio_count > 0:
                    router_metrics["mean_guide_ratio"] = float(guide_ratio_sum / guide_ratio_count)
                if scale_score_count > 0:
                    router_metrics["mean_scale_score"] = float(scale_score_sum / scale_score_count)

            results = {
                "split": split,
                "guide_w": guide_w,
                "MSE": nmse_total / evalpoints_total,
                "MAE": nmae_total / evalpoints_total,
                "horizon_metrics": horizon_metrics,
                "band_metrics": band_metrics,
            }
            if router_metrics:
                results["router_metrics"] = router_metrics
            if append_to_config_results and foldername:
                with open(os.path.join(foldername, "config_results.json"), "a") as f:
                    json.dump(results, f, indent=4)
            guide_tag = str(guide_w).replace("-", "m").replace(".", "p")
            if foldername:
                metrics_prefix = "eval" if split == "test" else split
                with open(os.path.join(foldername, f"{metrics_prefix}_metrics_guide_{guide_tag}.json"), "w") as f:
                    json.dump(results, f, indent=4)
            print("MSE:", nmse_total / evalpoints_total)
            print("MAE:", nmae_total / evalpoints_total)
            if horizon_metrics:
                print("Horizon metrics:", json.dumps(horizon_metrics, ensure_ascii=True))
            if band_metrics:
                print("Band metrics:", json.dumps(band_metrics, ensure_ascii=True))
            if router_metrics:
                print("Router metrics:", json.dumps(router_metrics, ensure_ascii=True))
    return nmse_total / evalpoints_total
