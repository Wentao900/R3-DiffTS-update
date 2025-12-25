import json
import re
from typing import Dict, Optional, Sequence

import numpy as np

_DEFAULT_FIELDS = {
    "direction": "flat",
    "strength": "moderate",
    "volatility": "medium",
}

_DIRECTION_MAP = {
    "up": ("up", "upward", "increase", "rising", "positive", "bull"),
    "down": ("down", "downward", "decrease", "falling", "negative", "bear"),
    "flat": ("flat", "stable", "steady", "unchanged", "neutral"),
}
_STRENGTH_MAP = {
    "weak": ("weak", "mild", "slight", "small"),
    "moderate": ("moderate", "medium", "mid", "average"),
    "strong": ("strong", "sharp", "steep", "large"),
}
_VOLATILITY_MAP = {
    "low": ("low", "stable", "smooth", "quiet"),
    "medium": ("medium", "moderate", "mid"),
    "high": ("high", "volatile", "noisy", "turbulent"),
}

_STRENGTH_VALUE = {"weak": 0.5, "moderate": 1.0, "strong": 1.5}
_VOLATILITY_VALUE = {"low": 0.0, "medium": 0.5, "high": 1.0}
_DIRECTION_VALUE = {"down": -1.0, "flat": 0.0, "up": 1.0}


def _normalize_label(value: Optional[str], mapping: Dict[str, Sequence[str]], default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in mapping:
        return text
    for key, aliases in mapping.items():
        if text == key:
            return key
        if any(alias in text for alias in aliases):
            return key
    return default


def _safe_json_load(raw: str) -> Optional[Dict[str, object]]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            cleaned = raw.replace("'", "\"")
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def parse_structured_cot(cot_text: str) -> Optional[Dict[str, str]]:
    if not cot_text:
        return None
    match = re.search(r"\{.*\}", cot_text, flags=re.DOTALL)
    if not match:
        return None
    payload = _safe_json_load(match.group(0))
    if not isinstance(payload, dict):
        return None
    direction = _normalize_label(payload.get("direction"), _DIRECTION_MAP, _DEFAULT_FIELDS["direction"])
    strength = _normalize_label(payload.get("strength"), _STRENGTH_MAP, _DEFAULT_FIELDS["strength"])
    volatility = _normalize_label(payload.get("volatility"), _VOLATILITY_MAP, _DEFAULT_FIELDS["volatility"])
    return {"direction": direction, "strength": strength, "volatility": volatility}


def infer_trend_fields(numeric_history: Sequence[float]) -> Dict[str, str]:
    arr = np.asarray(numeric_history, dtype=float).reshape(-1)
    if arr.size < 2:
        return _DEFAULT_FIELDS.copy()
    slope = (arr[-1] - arr[0]) / max(arr.size - 1, 1)
    std = float(np.std(arr))
    mean_abs = float(np.mean(np.abs(arr))) + 1e-6
    direction = "flat"
    if slope > 1e-6:
        direction = "up"
    elif slope < -1e-6:
        direction = "down"
    norm_slope = abs(slope) / (std + 1e-6)
    if norm_slope < 0.1:
        strength = "weak"
    elif norm_slope < 0.5:
        strength = "moderate"
    else:
        strength = "strong"
    vol_ratio = std / mean_abs
    if vol_ratio < 0.1:
        volatility = "low"
    elif vol_ratio < 0.3:
        volatility = "medium"
    else:
        volatility = "high"
    return {"direction": direction, "strength": strength, "volatility": volatility}


def build_trend_fields(cot_text: str, numeric_history: Sequence[float]) -> Dict[str, str]:
    parsed = parse_structured_cot(cot_text)
    if parsed is not None:
        return parsed
    return infer_trend_fields(numeric_history)


def trend_fields_to_vector(fields: Dict[str, str]) -> np.ndarray:
    direction = _normalize_label(fields.get("direction"), _DIRECTION_MAP, _DEFAULT_FIELDS["direction"])
    strength = _normalize_label(fields.get("strength"), _STRENGTH_MAP, _DEFAULT_FIELDS["strength"])
    volatility = _normalize_label(fields.get("volatility"), _VOLATILITY_MAP, _DEFAULT_FIELDS["volatility"])
    return np.array(
        [
            _DIRECTION_VALUE[direction],
            _STRENGTH_VALUE[strength],
            _VOLATILITY_VALUE[volatility],
        ],
        dtype=np.float32,
    )
