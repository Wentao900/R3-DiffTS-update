"""Text-protocol rules from 最优方案.md (C1–C5, C17).

No tuned constants. The token budget is the domain's existing ``text_max_length``
minus the two BERT special tokens [CLS] and [SEP] (Devlin et al., 2019). The
shift gap is the window's history length. A year token is a 4-digit number.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from utils.text_presence import text_is_absent

_YEAR = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def whitespace_tokens(text: str) -> List[str]:
    return str(text).split()


def bert_content_budget(text_max_length: int) -> int:
    """Content tokens that fit in ``text_max_length`` once [CLS] and [SEP] are reserved."""
    return max(int(text_max_length) - 2, 1)


def fill_token_budget(segments: Sequence[str], budget: int, tokenize=whitespace_tokens) -> str:
    """Concatenate newest-first segments until ``budget`` tokens are used.

    The last segment is cut so the budget is filled. There is no document-count cap.
    """
    budget = int(budget)
    if budget < 1:
        return ""
    chosen: List[str] = []
    used = 0
    for segment in segments:
        tokens = tokenize(segment)
        if not tokens:
            continue
        room = budget - used
        if room <= 0:
            break
        if len(tokens) <= room:
            chosen.append(" ".join(tokens))
            used += len(tokens)
            continue
        chosen.append(" ".join(tokens[:room]))
        used = budget
        break
    return " ".join(chosen)


def fact_mentions_future_year(fact: str, end_date) -> bool:
    """True when the fact names a 4-digit year later than its own end_date year.

    This is the year rule in 最优方案 §1.4. A 4-digit number is treated as a year
    token. The false-positive rate on non-year numbers is not estimated here.
    """
    if fact is None or (isinstance(fact, float) and pd.isna(fact)):
        return False
    year = int(pd.Timestamp(end_date).year)
    for match in _YEAR.findall(str(fact)):
        if int(match) > year:
            return True
    return False


def drop_future_year_rows(frame: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, List[dict]]:
    """Drop rows whose fact mentions a year after that row's end_date. Return the audit rows."""
    if frame is None or frame.empty or "fact" not in frame.columns or "end_date" not in frame.columns:
        return frame, []
    removed = []
    keep = []
    for index, row in frame.iterrows():
        fact = row["fact"]
        if fact_mentions_future_year(fact, row["end_date"]):
            removed.append(
                {
                    "source": source,
                    "row_index": int(index) if not isinstance(index, tuple) else index,
                    "start_date": str(row["start_date"]) if "start_date" in frame.columns else "",
                    "end_date": str(pd.Timestamp(row["end_date"]).date()),
                    "fact": str(fact),
                }
            )
            keep.append(False)
        else:
            keep.append(True)
    filtered = frame.loc[keep].reset_index(drop=True)
    return filtered, removed


def report_fact_is_placeholder(fact) -> bool:
    """NA and NA-plus-explanation. Dates and the domain description are not facts."""
    return text_is_absent(fact, placeholder_absent=True)


def strip_na_tokens(text: str) -> str:
    """Drop standalone NA sentinels from a retrieval query. Other tokens stay."""
    kept = []
    for token in str(text).split():
        if token.strip().upper() == "NA":
            continue
        kept.append(token)
    return " ".join(kept)


def decide_text_sources(raw_marks, ret_marks, cot_marks=None) -> dict:
    """R1 and R2 on one training split.

    R1: a source that is never present is not used.
    R2: beta_s is kept only when, among windows with any source present, m_s is
    both 0 and 1. Otherwise beta_s * m_s is constant on those windows and is
    absorbed by b.
    """
    raw = [float(value) > 0 for value in raw_marks]
    ret = [float(value) > 0 for value in ret_marks]
    names = ["raw", "ret"]
    columns = [raw, ret]
    if cot_marks is not None:
        names.append("cot")
        columns.append([float(value) > 0 for value in cot_marks])
    any_present = [any(flags) for flags in zip(*columns)]
    used = []
    dropped = []
    beta_keep = []
    for name, column in zip(names, columns):
        if not any(column):
            dropped.append(name)
            used.append(False)
            beta_keep.append(False)
            continue
        used.append(True)
        present = [flag for flag, active in zip(column, any_present) if active]
        beta_keep.append(any(present) and not all(present))
    n_beta = int(sum(beta_keep))
    return {
        "sources": names,
        "used": used,
        "dropped": dropped,
        "beta_keep": beta_keep,
        "n_gate_parameters": 2 + n_beta,
    }


def cyclic_source_shift(indices: Sequence[int], min_gap: int) -> dict:
    """Cyclic shift inside one source's present windows.

    The shift is the offset that lets the most windows move by at least
    ``min_gap`` rows (the history length). Ties maximize the smallest gap.
    Windows that still cannot clear ``min_gap`` stay put.
    """
    positions = [int(value) for value in indices]
    n = len(positions)
    if n < 2:
        return {
            "shift": 0,
            "donor": list(range(n)),
            "shifted": [False] * n,
            "min_gap": None,
            "unshifted": n,
        }
    best = None
    for offset in range(1, n):
        gaps = []
        ok = 0
        for i in range(n):
            donor_pos = positions[(i + offset) % n]
            gap = abs(donor_pos - positions[i])
            gaps.append(gap)
            if gap >= int(min_gap):
                ok += 1
        candidate = (ok, min(gaps), offset)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    offset = best[2]
    donor = []
    shifted = []
    achieved = []
    for i in range(n):
        gap = abs(positions[(i + offset) % n] - positions[i])
        if gap >= int(min_gap):
            donor.append((i + offset) % n)
            shifted.append(True)
            achieved.append(gap)
        else:
            donor.append(i)
            shifted.append(False)
    return {
        "shift": int(offset),
        "donor": donor,
        "shifted": shifted,
        "min_gap": None if not achieved else int(min(achieved)),
        "unshifted": int(sum(1 for flag in shifted if not flag)),
    }
