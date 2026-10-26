"""Inference-only overrides for a trained checkpoint.

These stay out of exe_forecasting.py because importing that script parses
argv and starts a run.
"""


def parse_infer_override(text):
    spec = {}
    for part in str(text or "").split(","):
        if not part.strip() or "=" not in part:
            continue
        key, value = part.split("=", 1)
        spec[key.strip()] = value.strip()
    return spec


def apply_infer_override(model, loaders, spec):
    if not spec:
        return
    if "guide_w" in spec:
        model.infer_guide_w = float(spec["guide_w"])
    if "text_gate_mode" in spec:
        model.text_gate_mode = str(spec["text_gate_mode"]).lower()
    if "gate_constant" in spec:
        model.infer_gate_constant = float(spec["gate_constant"])
        model.text_gate_mode = str(spec.get("text_gate_mode", "constant")).lower()
    if "mask_sources" in spec:
        model.infer_mask_sources = [item.strip() for item in spec["mask_sources"].split("+") if item.strip()]
    if spec.get("text_shift_mode") == "cyclic_per_source":
        for loader in loaders:
            dataset = getattr(loader, "dataset", None)
            if dataset is None or not hasattr(dataset, "_prepare_text_shift"):
                continue
            dataset.text_shift_mode = "cyclic_per_source"
            dataset.text_shift_splits = [dataset.split_name]
            dataset._prepare_text_shift()
    if getattr(model, "log_control_calls", False):
        model.control_log["infer_override"] = model.control_log.get("infer_override", 0) + 1
    print("[infer_override]", spec)
