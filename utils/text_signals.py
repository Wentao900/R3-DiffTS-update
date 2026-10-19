"""Raw text-signal helpers.

These functions repair degenerate text features (substring keywords, unread CoT
JSON, cross-column slopes, left-truncation of the composed string). They do
not score validity and they do not multiply hand-tuned trust weights.
"""

import re
from typing import Dict, Optional, Sequence

import numpy as np

from utils.trend_prior import parse_structured_cot

_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-z]+)?|\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Surface forms only. Matching is on word boundaries, so "up" does not fire
# inside "supply" and "fall" does not fire inside "rainfall".
_DIRECTION_UP = ("increase", "rise", "rising", "upward", "higher", "bull", "up")
_DIRECTION_DOWN = ("decrease", "decline", "fall", "falling", "downward", "lower", "bear", "down")
_STRENGTH_WEAK = ("slight", "mild", "small", "weak")
_STRENGTH_STRONG = ("sharp", "strong", "steep", "large")
_VOL_LOW = ("stable", "steady", "smooth", "quiet")
_VOL_HIGH = ("volatile", "turbulent", "noisy", "uncertain")

_DIRECTION_VALUE = {"down": -1.0, "flat": 0.0, "up": 1.0}
_STRENGTH_ORDINAL = {"weak": 0.0, "moderate": 1.0, "strong": 2.0}
_VOLATILITY_ORDINAL = {"low": 0.0, "medium": 1.0, "high": 2.0}

# Space reserved inside the single left-truncated string so each block starts
# inside a short encoder window. Not a validity weight. Learned mode encodes
# each source with its own text_max_length and does not use these caps.
COMPOSE_BLOCK_CHARS = {
    "hypothesis": 160,
    "summary": 120,
    "evidence_item": 120,
    "raw": 480,
}

# Per source: available, token_count, content_ratio, unique_ratio,
# direction (-1/0/1), strength ordinal (0/1/2), volatility ordinal (0/1/2).
PER_SOURCE_FEATURE_DIM = 7
# Shared, unmultiplied: numeric direction, |slope|, std, recency days,
# report count, recent report count, retrieval novelty, cot novelty.
SHARED_FEATURE_DIM = 8
VALIDITY_FEATURE_DIM = PER_SOURCE_FEATURE_DIM * 3 + SHARED_FEATURE_DIM

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at", "by",
    "with", "from", "is", "are", "was", "were", "be", "been", "this", "that",
    "these", "those", "as", "it", "its", "their", "his", "her", "they", "we",
    "you", "our", "into", "over", "after", "before", "during", "about",
}


def contains_term(text: str, term: str) -> bool:
    """True when `term` occurs as a whole token, not as a substring."""
    if not text or not term:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def contains_any_term(text: str, terms: Sequence[str]) -> bool:
    return any(contains_term(text, term) for term in terms)


def _normalize_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if text.upper() == "NA":
        return ""
    return text


def _keyword_trend_fields(text: str) -> Optional[Dict[str, str]]:
    normalized = _normalize_text(text).lower()
    if not normalized:
        return None
    direction = "flat"
    if contains_any_term(normalized, _DIRECTION_UP):
        direction = "up"
    elif contains_any_term(normalized, _DIRECTION_DOWN):
        direction = "down"
    strength = "moderate"
    if contains_any_term(normalized, _STRENGTH_WEAK):
        strength = "weak"
    elif contains_any_term(normalized, _STRENGTH_STRONG):
        strength = "strong"
    volatility = "medium"
    # Low is checked first so a phrase like "smooth ... volatility" is not
    # forced high merely because the word "volatility" appears.
    if contains_any_term(normalized, _VOL_LOW) or contains_term(normalized, "low volatility"):
        volatility = "low"
    elif contains_any_term(normalized, _VOL_HIGH) or contains_term(normalized, "volatility"):
        volatility = "high"
    return {"direction": direction, "strength": strength, "volatility": volatility}


def infer_text_trend_fields(text: str) -> Optional[Dict[str, str]]:
    """Read a structured CoT JSON object when one is present.

    Keyword scan is only the fallback. Scanning the JSON source itself made
    direction stay "flat" (the values "up"/"down" were not in the keyword
    list) and volatility stay "high" (the key name "volatility" matched).
    """
    normalized = _normalize_text(text)
    if not normalized:
        return None
    parsed = parse_structured_cot(normalized)
    if parsed is not None:
        return parsed
    return _keyword_trend_fields(normalized)


def target_series(numeric_history, target_index: int = 0) -> np.ndarray:
    """Return the target column of a window.

    Windows are (time, feature). Flattening that matrix and taking the last
    minus the first element subtracts across columns. Column 0 is the target
    when covariates are appended after it.
    """
    arr = np.asarray(numeric_history, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim >= 2:
        arr = arr.reshape(arr.shape[0], -1)
        idx = int(target_index)
        if idx < 0 or idx >= arr.shape[1]:
            idx = 0
        return np.asarray(arr[:, idx], dtype=float).reshape(-1)
    return arr.reshape(-1)


def numeric_endpoint_stats(numeric_history, target_index: int = 0) -> Dict[str, float]:
    arr = target_series(numeric_history, target_index=target_index)
    if arr.size == 0:
        return {"slope": 0.0, "std": 0.0, "mean_abs": 0.0, "mean": 0.0, "last": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    mean_abs = float(np.mean(np.abs(arr)))
    slope = float(arr[-1] - arr[0]) / max(arr.size - 1, 1)
    return {
        "slope": slope,
        "std": std,
        "mean_abs": mean_abs,
        "mean": mean,
        "last": float(arr[-1]),
    }


def numeric_trend_fields_from_series(arr: np.ndarray) -> Dict[str, str]:
    """Discretize a 1-d target series. Thresholds match the previous numeric prior."""
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size < 2:
        return {"direction": "flat", "strength": "moderate", "volatility": "medium"}
    slope = float((arr[-1] - arr[0]) / max(arr.size - 1, 1))
    std = float(np.std(arr))
    mean_abs = float(np.mean(np.abs(arr))) + 1e-6
    if slope > 1e-6:
        direction = "up"
    elif slope < -1e-6:
        direction = "down"
    else:
        direction = "flat"
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


def clip_block(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return cleaned[: limit - 3].rstrip() + "..."


def compose_priority_text(
    raw_text: str = "",
    numeric_summary: str = "",
    trend_hypothesis: str = "",
    retrieved: Optional[Sequence[str]] = None,
) -> str:
    """Put hypothesis, summary, and evidence before raw text.

    A left-truncating encoder then sees those blocks. Raw text fills whatever
    budget remains and is the part that may be cut.
    """
    retrieved = list(retrieved or [])
    blocks = []
    hypothesis = clip_block(trend_hypothesis, COMPOSE_BLOCK_CHARS["hypothesis"])
    summary = clip_block(numeric_summary, COMPOSE_BLOCK_CHARS["summary"])
    if hypothesis:
        blocks.append("[TREND HYPOTHESIS]\n" + hypothesis)
    if summary:
        blocks.append("[NUMERICAL SUMMARY]\n" + summary)
    evidence_lines = []
    for idx, item in enumerate(retrieved, start=1):
        clipped = clip_block(item, COMPOSE_BLOCK_CHARS["evidence_item"])
        if clipped:
            evidence_lines.append(f"{idx}) {clipped}")
    if evidence_lines:
        blocks.append("[RETRIEVED EVIDENCE]\n" + "\n".join(evidence_lines))
    raw = clip_block(raw_text, COMPOSE_BLOCK_CHARS["raw"]) if _normalize_text(raw_text) else ""
    if raw:
        blocks.append("[RAW TEXT]\n" + raw)
    return "\n\n".join(blocks)


def order_recent_segments(desc: Optional[str], segments: Sequence[str], token_budget: int) -> str:
    """Keep the newest segments inside a word budget and emit newest-first.

    The previous order was oldest-first, so a left-truncating encoder saw the
    domain description and the oldest report.
    """
    cleaned = [str(segment).strip() for segment in segments if str(segment).strip()]
    budget = max(int(token_budget), 1)
    desc_text = desc.strip() if isinstance(desc, str) else ""
    if cleaned and desc_text and cleaned[0] == desc_text:
        body = cleaned[1:]
    else:
        body = cleaned
        if not desc_text and cleaned:
            desc_text = ""
    desc_tokens = desc_text.split() if desc_text else []
    if len(desc_tokens) >= budget:
        return " ".join(desc_tokens[:budget])
    remaining = budget - len(desc_tokens)
    selected_newest_first = []
    used = 0
    for segment in reversed(body):
        segment_tokens = segment.split()
        if not segment_tokens:
            continue
        if used + len(segment_tokens) > remaining:
            if not selected_newest_first and remaining > 0:
                selected_newest_first.append(" ".join(segment_tokens[-remaining:]))
            break
        selected_newest_first.append(segment)
        used += len(segment_tokens)
    parts = []
    if desc_text:
        parts.append(desc_text)
    parts.extend(selected_newest_first)
    if not parts:
        return "NA"
    return " ".join(parts)


def _token_list(text: str):
    normalized = _normalize_text(text).lower()
    if not normalized:
        return []
    return _TOKEN_RE.findall(normalized)


def source_surface_features(text: str) -> np.ndarray:
    """Unmultiplied surface features for one text source. Empty text is zeros."""
    tokens = _token_list(text)
    fields = infer_text_trend_fields(text) if tokens else None
    if not tokens or fields is None:
        return np.zeros((PER_SOURCE_FEATURE_DIM,), dtype=np.float32)
    content = [token for token in tokens if token not in _STOPWORDS]
    unique_ratio = len(set(tokens)) / float(len(tokens))
    content_ratio = len(content) / float(len(tokens))
    return np.asarray(
        [
            1.0,
            float(len(tokens)),
            float(content_ratio),
            float(unique_ratio),
            float(_DIRECTION_VALUE[fields["direction"]]),
            float(_STRENGTH_ORDINAL[fields["strength"]]),
            float(_VOLATILITY_ORDINAL[fields["volatility"]]),
        ],
        dtype=np.float32,
    )


def _jaccard_distance(text_a: str, text_b: str) -> float:
    tokens_a = set(_WORD_RE.findall(_normalize_text(text_a).lower()))
    tokens_b = set(_WORD_RE.findall(_normalize_text(text_b).lower()))
    if not tokens_a and not tokens_b:
        return 0.0
    union = len(tokens_a | tokens_b)
    if union <= 0:
        return 0.0
    return float(1.0 - (len(tokens_a & tokens_b) / float(union)))


def build_raw_validity_features(
    raw_text: str,
    ret_text: str,
    cot_text: str,
    seq_x,
    recency_days,
    report_count: float,
    recent_report_count: float,
    target_index: int = 0,
) -> np.ndarray:
    """Feature vector for the learned gate.

    Nothing here is a product of trust, alignment, or hand-tuned gates.
    `recency_days` is the raw age, not exp(-age / tau).
    """
    parts = [
        source_surface_features(raw_text),
        source_surface_features(ret_text),
        source_surface_features(cot_text),
    ]
    stats = numeric_endpoint_stats(seq_x, target_index=target_index)
    numeric_fields = numeric_trend_fields_from_series(target_series(seq_x, target_index))
    if recency_days is None:
        age = 0.0
    else:
        age = float(max(recency_days, 0.0))
    shared = np.asarray(
        [
            float(_DIRECTION_VALUE[numeric_fields["direction"]]),
            float(abs(stats["slope"])),
            float(stats["std"]),
            age,
            float(max(report_count, 0.0)),
            float(max(recent_report_count, 0.0)),
            _jaccard_distance(raw_text, ret_text),
            _jaccard_distance(raw_text, cot_text),
        ],
        dtype=np.float32,
    )
    vector = np.concatenate(parts + [shared], axis=0).astype(np.float32)
    if vector.shape[0] != VALIDITY_FEATURE_DIM:
        raise RuntimeError(f"validity feature dim {vector.shape[0]} != {VALIDITY_FEATURE_DIM}")
    return vector
