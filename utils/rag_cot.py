import json
import re
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclass
class RAGCoTConfig:
    use_retrieval: bool = True
    top_k: int = 3
    max_new_tokens: int = 96
    temperature: float = 0.7
    cot_model: Optional[str] = None
    cache_size: int = 1024
    local_files_only: bool = True
    device: Optional[str] = None
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    structured_output: bool = False
    include_cot_in_text: bool = True
    use_two_stage_rag: bool = False
    rag_stage1_topk: int = -1
    rag_stage2_topk: int = -1
    two_stage_gate: bool = True
    trend_slope_eps: float = 1e-3
    debug: bool = False


class RAGCoTPipeline:
    """
    Lightweight retrieval + chain-of-thought text synthesizer.
    Builds a TF-IDF retriever over the domain search corpus and optionally
    calls a local causal LM to turn retrieved evidence + numeric stats into
    a short reasoning snippet that can be encoded by the text encoder.
    """

    def __init__(
        self,
        domain: str,
        search_df: Optional[pd.DataFrame],
        desc: str,
        lookback_len: int,
        pred_len: int,
        config: Optional[RAGCoTConfig] = None,
    ) -> None:
        self.domain = domain
        self.desc = desc
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.config = config or RAGCoTConfig()
        self.use_two_stage_rag = bool(self.config.use_two_stage_rag)
        self.two_stage_gate = bool(self.config.two_stage_gate)
        self.trend_slope_eps = float(self.config.trend_slope_eps)
        self.debug = bool(self.config.debug)
        self.rag_stage2_topk = self._resolve_stage2_topk(self.config.rag_stage2_topk)
        self.rag_stage1_topk = self._resolve_stage1_topk(self.config.rag_stage1_topk, self.rag_stage2_topk)
        self.search_df = self._prep_search_df(search_df)
        self.retriever = self._fit_retriever(self.search_df)
        self.generator = self._init_generator(self.config)
        self.cache: OrderedDict[str, Dict[str, str]] = OrderedDict()
        # Text budget (approx. whitespace tokens) to reduce harmful truncation by BERT.
        # Priority: NUMERICAL SUMMARY > TREND HYPOTHESIS/CoT > RETRIEVED EVIDENCE > RAW TEXT.
        # Note: BERT still truncates at its own tokenizer max length; this makes truncation predictable.
        self._text_budget_total_words = 480
        self._text_budget_num_words = 80
        self._text_budget_trend_words = 120
        self._text_budget_evidence_words = 240
        self._text_budget_raw_words = 240

    def _truncate_words(self, text: str, max_words: int) -> str:
        if not text or max_words <= 0:
            return ""
        words = " ".join(str(text).split()).split(" ")
        if len(words) <= max_words:
            return " ".join(words).strip()
        return " ".join(words[:max_words]).strip()

    def _apply_section_budgets(
        self,
        raw_text: str,
        numeric_summary: str,
        trend_text: str,
        evidence_items: Sequence[str],
    ) -> Tuple[str, str, str, List[str]]:
        numeric_summary = self._truncate_words(numeric_summary, self._text_budget_num_words)
        trend_text = self._truncate_words(trend_text, self._text_budget_trend_words)
        evidence_joined = " ".join([self._truncate_evidence(item) for item in (evidence_items or [])]).strip()
        evidence_joined = self._truncate_words(evidence_joined, self._text_budget_evidence_words)
        raw_text = self._truncate_words(raw_text, self._text_budget_raw_words)

        # Enforce total budget by trimming low-priority sections first.
        def count_words(t: str) -> int:
            return 0 if not t else len(str(t).split())

        total = count_words(numeric_summary) + count_words(trend_text) + count_words(evidence_joined) + count_words(raw_text)
        if total > self._text_budget_total_words:
            overflow = total - self._text_budget_total_words
            # Trim RAW first
            raw_keep = max(0, count_words(raw_text) - overflow)
            raw_text = self._truncate_words(raw_text, raw_keep)
            total = count_words(numeric_summary) + count_words(trend_text) + count_words(evidence_joined) + count_words(raw_text)
        if total > self._text_budget_total_words:
            overflow = total - self._text_budget_total_words
            # Then trim evidence
            ev_keep = max(0, count_words(evidence_joined) - overflow)
            evidence_joined = self._truncate_words(evidence_joined, ev_keep)
            total = count_words(numeric_summary) + count_words(trend_text) + count_words(evidence_joined) + count_words(raw_text)
        if total > self._text_budget_total_words:
            overflow = total - self._text_budget_total_words
            # Then trim trend
            tr_keep = max(0, count_words(trend_text) - overflow)
            trend_text = self._truncate_words(trend_text, tr_keep)
            total = count_words(numeric_summary) + count_words(trend_text) + count_words(evidence_joined) + count_words(raw_text)
        if total > self._text_budget_total_words:
            overflow = total - self._text_budget_total_words
            # Finally trim numeric summary (should rarely happen)
            num_keep = max(0, count_words(numeric_summary) - overflow)
            numeric_summary = self._truncate_words(numeric_summary, num_keep)

        evidence_items_out = [evidence_joined] if evidence_joined else []
        return raw_text, numeric_summary, trend_text, evidence_items_out

    def _resolve_stage2_topk(self, stage2_topk: int) -> int:
        if stage2_topk and stage2_topk > 0:
            return int(stage2_topk)
        return max(int(self.config.top_k), 0)

    def _resolve_stage1_topk(self, stage1_topk: int, stage2_topk: int) -> int:
        if stage1_topk and stage1_topk > 0:
            return int(stage1_topk)
        derived = max(stage2_topk * 3, stage2_topk)
        return min(20, derived) if derived > 0 else 0

    def _prep_search_df(self, search_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if search_df is None:
            return pd.DataFrame(columns=["fact", "start_date", "end_date"])
        df = search_df.copy()
        df["fact"] = df["fact"].fillna("").astype(str)
        if "start_date" in df.columns and not np.issubdtype(df["start_date"].dtype, np.datetime64):
            df["start_date"] = pd.to_datetime(df["start_date"])
        if "end_date" in df.columns and not np.issubdtype(df["end_date"].dtype, np.datetime64):
            df["end_date"] = pd.to_datetime(df["end_date"])
        return df

    def _fit_retriever(self, search_df: pd.DataFrame) -> Optional[Dict[str, object]]:
        if search_df.empty or not self.config.use_retrieval:
            return None
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4096, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(search_df["fact"].tolist())
        return {"vectorizer": vectorizer, "matrix": matrix}

    def _init_generator(self, config: RAGCoTConfig):
        if config.cot_model is None:
            return None
        device_index = self._resolve_device_index(config.device)
        trust_remote = config.trust_remote_code or (config.cot_model and "qwen" in config.cot_model.lower())
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.cot_model,
                local_files_only=config.local_files_only,
                trust_remote_code=trust_remote,
            )
            model_kwargs = {
                "local_files_only": config.local_files_only,
                "trust_remote_code": trust_remote,
            }
            if config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            if config.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            model = AutoModelForCausalLM.from_pretrained(
                config.cot_model,
                **model_kwargs,
            )
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_index,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Falling back to template CoT because model '{config.cot_model}' "
                f"could not be loaded locally ({exc})."
            )
            print(
                f"[RAGCoT] CoT model load failed for '{config.cot_model}': {exc}",
                file=sys.stderr,
            )
            return None

    def _resolve_device_index(self, device: Optional[str]) -> int:
        if device is None:
            return 0 if torch.cuda.is_available() else -1
        if device.startswith("cuda") and torch.cuda.is_available():
            parts = device.split(":")
            return int(parts[1]) if len(parts) > 1 else 0
        return -1

    def _retrieve(self, query_text: str, top_k: Optional[int] = None) -> List[str]:
        k = self.config.top_k if top_k is None else int(top_k)
        if not self.retriever or not self.config.use_retrieval or k <= 0:
            return []
        query_vec = self.retriever["vectorizer"].transform([query_text])
        sims = cosine_similarity(query_vec, self.retriever["matrix"]).ravel()
        if sims.size == 0:
            return []
        top_idx = sims.argsort()[::-1][:k]
        return [
            self.search_df.iloc[i].fact
            for i in top_idx
            if sims[i] > 0 and len(self.search_df.iloc[i].fact.strip()) > 0
        ]

    def _summarize_numeric(self, numeric_history: Sequence[float]) -> str:
        arr = np.asarray(numeric_history, dtype=float).flatten()
        if arr.size == 0:
            return "No numeric history available."
        last = arr[-1]
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        slope = float(arr[-1] - arr[0]) / max(arr.size - 1, 1)
        direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"
        return (
            f"last={last:.4f}, mean={mean:.4f}, std={std:.4f}, "
            f"trend is {direction} with slope {slope:.4f}"
        )

    def _compute_numeric_stats(self, numeric_history: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(numeric_history, dtype=float).flatten()
        if arr.size == 0:
            return {
                "slope": 0.0,
                "std": 0.0,
                "mean_abs": 0.0,
                "mean": 0.0,
            }
        mean = float(np.mean(arr))
        slope = float(arr[-1] - arr[0]) / max(arr.size - 1, 1)
        std = float(np.std(arr))
        mean_abs = float(np.mean(np.abs(arr)))
        return {"slope": slope, "std": std, "mean_abs": mean_abs, "mean": mean}

    def _format_prompt(
        self,
        numeric_summary: str,
        retrieved: List[str],
    ) -> str:
        evidence = "\n".join([f"- {item}" for item in retrieved]) if retrieved else "No extra evidence."
        if self.config.structured_output:
            return (
                f"{self.desc}\n"
                f"Historical summary (lookback {self.lookback_len}): {numeric_summary}\n"
                f"Retrieved evidence:\n{evidence}\n"
                "Return only a JSON object with keys: "
                'direction ("up"|"down"|"flat"), '
                'strength ("weak"|"moderate"|"strong"), '
                'volatility ("low"|"medium"|"high"), '
                'reasoning (short explanation).'
            )
        return (
            f"{self.desc}\n"
            f"Historical summary (lookback {self.lookback_len}): {numeric_summary}\n"
            f"Retrieved evidence:\n{evidence}\n"
            f"Reason step by step to sketch an intermediate trend for the next {self.pred_len} steps "
            f"before a final forecast."
        )

    def _format_trend_prompt(self, numeric_summary: str, retrieved: List[str]) -> str:
        evidence = "\n".join([f"- {item}" for item in retrieved]) if retrieved else "No extra evidence."
        return (
            f"{self.desc}\n"
            f"Historical summary (lookback {self.lookback_len}): {numeric_summary}\n"
            f"Retrieved evidence:\n{evidence}\n"
            "Return only a compact JSON object with keys: "
            'direction ("up"|"down"|"flat"), '
            'strength ("weak"|"moderate"|"strong"), '
            'volatility ("low"|"medium"|"high"), '
            'key_factors (short phrase).'
        )

    def _fallback_cot(self, numeric_summary: str, retrieved: List[str]) -> str:
        steps = [
            f"1) Summarize numeric window: {numeric_summary}.",
        ]
        if retrieved:
            steps.append(f"2) Align with retrieved signals: {' '.join(retrieved[:2])}.")
        steps.append(
            "3) Extrapolate a smooth intermediate trend that respects the direction and volatility, "
            "without giving exact predictions."
        )
        return " ".join(steps)

    def _fallback_trend_hypothesis(self, numeric_history: Sequence[float]) -> str:
        stats = self._compute_numeric_stats(numeric_history)
        slope = stats["slope"]
        direction = "flat"
        if slope > self.trend_slope_eps:
            direction = "up"
        elif slope < -self.trend_slope_eps:
            direction = "down"
        std = stats["std"]
        mean_abs = stats["mean_abs"]
        norm_slope = abs(slope) / (std + 1e-6)
        if norm_slope < 0.1:
            strength = "weak"
        elif norm_slope < 0.5:
            strength = "moderate"
        else:
            strength = "strong"
        vol_ratio = std / (mean_abs + 1e-6)
        if vol_ratio < 0.1:
            volatility = "low"
        elif vol_ratio < 0.3:
            volatility = "medium"
        else:
            volatility = "high"
        return (
            '{'
            f'"direction":"{direction}",'
            f'"strength":"{strength}",'
            f'"volatility":"{volatility}",'
            '"key_factors":"numeric history"'
            '}'
        )

    def _format_stats_hint(self, numeric_stats: Dict[str, float]) -> str:
        slope = numeric_stats.get("slope", 0.0)
        std = numeric_stats.get("std", 0.0)
        mean = numeric_stats.get("mean", 0.0)
        return f"slope={slope:.4f}, std={std:.4f}, mean={mean:.4f}"

    def _augment_trend_hypothesis(self, trend_hypothesis: str, numeric_stats: Dict[str, float]) -> str:
        stats_hint = self._format_stats_hint(numeric_stats)
        raw = self._extract_json_block(trend_hypothesis)
        try:
            payload = json.loads(raw)
        except Exception:
            if not trend_hypothesis:
                return f"stats: {stats_hint}"
            return f"{trend_hypothesis} | stats: {stats_hint}"
        if not isinstance(payload, dict):
            return f"{trend_hypothesis} | stats: {stats_hint}"
        key_factors = str(payload.get("key_factors", "")).strip()
        payload["key_factors"] = f"{key_factors}; {stats_hint}".strip("; ").strip()
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    def _generate_cot(self, prompt: str, numeric_summary: str, retrieved: List[str]) -> str:
        if self.generator is None:
            return self._fallback_cot(numeric_summary, retrieved)
        try:
            output = self.generator(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                num_return_sequences=1,
                do_sample=True,
            )
            text = output[0]["generated_text"]
            return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Falling back to template CoT because generation failed ({exc}).")
            return self._fallback_cot(numeric_summary, retrieved)

    def _extract_json_block(self, text: str) -> str:
        if not text:
            return ""
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return match.group(0).strip()
        return text.strip()

    def _generate_trend_hypothesis(
        self,
        prompt: str,
        numeric_summary: str,
        retrieved: List[str],
        numeric_history: Sequence[float],
    ) -> str:
        if self.generator is None:
            return self._fallback_trend_hypothesis(numeric_history)
        raw = self._generate_cot(prompt, numeric_summary, retrieved)
        if raw.startswith("1) Summarize numeric window"):
            return self._fallback_trend_hypothesis(numeric_history)
        cleaned = self._extract_json_block(raw)
        return cleaned if cleaned else self._fallback_trend_hypothesis(numeric_history)

    def _is_empty_text(self, text: Optional[str]) -> bool:
        if text is None:
            return True
        stripped = text.strip()
        return stripped == "" or stripped.upper() == "NA"

    def _build_query(self, numeric_summary: str, base_text: str) -> str:
        return f"{self.domain} {numeric_summary} {base_text}"

    def _build_stage2_query(self, base_query: str, trend_hypothesis: str) -> str:
        return (
            f"{base_query}\n"
            "[TREND HYPOTHESIS]\n"
            f"{trend_hypothesis}\n"
            "Retrieve evidence that best supports/explains this trend hypothesis and is useful for forecasting."
        )

    def _trend_hypothesis_to_query_text(self, trend_hypothesis: str) -> str:
        raw = self._extract_json_block(trend_hypothesis)
        try:
            payload = json.loads(raw)
        except Exception:
            return trend_hypothesis
        if not isinstance(payload, dict):
            return trend_hypothesis
        direction = str(payload.get("direction", "flat")).strip().lower()
        strength = str(payload.get("strength", "moderate")).strip().lower()
        volatility = str(payload.get("volatility", "medium")).strip().lower()
        key_factors = str(payload.get("key_factors", "")).strip()
        direction_text = {
            "up": "upward",
            "down": "downward",
            "flat": "flat",
        }.get(direction, direction)
        parts = [f"Likely {direction_text} trend", f"{strength} strength", f"{volatility} volatility"]
        if key_factors:
            parts.append(f"key factors: {key_factors}")
        return ", ".join(parts)

    def _merge_retrieved(self, primary: List[str], fallback: List[str], max_items: int) -> List[str]:
        seen = set()
        merged = []
        for item in primary + fallback:
            key = item.strip()
            if not key or key in seen:
                continue
            merged.append(item)
            seen.add(key)
            if max_items > 0 and len(merged) >= max_items:
                break
        return merged

    def _truncate_evidence(self, text: str, max_chars: int = 400) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 3].rstrip() + "..."

    def _compose_one_shot_text(
        self,
        base_text: str,
        numeric_summary: str,
        retrieved: List[str],
        cot_text: str,
    ) -> str:
        raw_text_in = base_text if (base_text and base_text != "NA") else "NA"
        trend_text_in = cot_text if (cot_text and self.config.include_cot_in_text) else ""
        raw_text, numeric_summary, trend_text, retrieved_budgeted = self._apply_section_budgets(
            raw_text=("" if raw_text_in == "NA" else raw_text_in),
            numeric_summary=numeric_summary,
            trend_text=trend_text_in,
            evidence_items=retrieved or [],
        )
        raw_text = raw_text if raw_text else "NA"
        lines = [
            "[NUMERICAL SUMMARY]",
            numeric_summary,
            "",
            "[TREND HYPOTHESIS]" if self.config.structured_output else "[TREND]",
            trend_text if trend_text else "NA",
            "",
            "[RETRIEVED EVIDENCE]",
        ]
        if retrieved_budgeted:
            for idx, item in enumerate(retrieved_budgeted, start=1):
                lines.append(f"{idx}) {item}")
        else:
            lines.append("1) NA")
        lines.extend(["", "[RAW TEXT]", raw_text])
        return "\n".join(lines)

    def _compose_two_stage_text(
        self,
        base_text: str,
        numeric_summary: str,
        trend_hypothesis: str,
        retrieved: List[str],
    ) -> str:
        raw_text_in = base_text if not self._is_empty_text(base_text) else "NA"
        raw_text, numeric_summary, trend_hypothesis, retrieved_budgeted = self._apply_section_budgets(
            raw_text=("" if raw_text_in == "NA" else raw_text_in),
            numeric_summary=numeric_summary,
            trend_text=trend_hypothesis,
            evidence_items=retrieved or [],
        )
        raw_text = raw_text if raw_text else "NA"
        lines = [
            "[NUMERICAL SUMMARY]",
            numeric_summary,
            "",
            "[TREND HYPOTHESIS]",
            trend_hypothesis,
            "",
            "[RETRIEVED EVIDENCE - REFINED]",
        ]
        if retrieved_budgeted:
            # Keep the fixed template, but the evidence content is already budgeted/prioritized.
            for idx, item in enumerate(retrieved_budgeted, start=1):
                lines.append(f"{idx}) {item}")
        else:
            lines.append("1) NA")
        lines.extend(
            [
                "",
                "[RAW TEXT]",
                raw_text,
            ]
        )
        return "\n".join(lines)

    def _build_one_shot_guidance(
        self,
        numeric_history: Sequence[float],
        base_text: str,
    ) -> Dict[str, str]:
        numeric_summary = self._summarize_numeric(numeric_history)
        query = self._build_query(numeric_summary, base_text)
        retrieved = self._retrieve(query)
        prompt = self._format_prompt(numeric_summary, retrieved)
        cot_text = self._generate_cot(prompt, numeric_summary, retrieved)
        composed_text = self._compose_one_shot_text(base_text, numeric_summary, retrieved, cot_text)
        return {
            "cot_text": cot_text,
            "retrieved_text": " ".join(retrieved),
            "composed_text": composed_text,
        }

    def build_guidance_text(
        self,
        numeric_history: Sequence[float],
        start_date,
        end_date,
        base_text: str,
    ) -> Dict[str, str]:
        cache_key = f"{start_date}-{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.use_two_stage_rag:
            packaged = self._build_one_shot_guidance(numeric_history, base_text)
            self.cache[cache_key] = packaged
            if len(self.cache) > self.config.cache_size:
                self.cache.popitem(last=False)
            return packaged

        numeric_summary = self._summarize_numeric(numeric_history)
        numeric_stats = self._compute_numeric_stats(numeric_history)
        query = self._build_query(numeric_summary, base_text)

        if self.two_stage_gate and self._is_empty_text(base_text) and abs(numeric_stats["slope"]) < self.trend_slope_eps:
            packaged = self._build_one_shot_guidance(numeric_history, base_text)
            self.cache[cache_key] = packaged
            if len(self.cache) > self.config.cache_size:
                self.cache.popitem(last=False)
            return packaged

        retrieved_stage1 = self._retrieve(query, top_k=self.rag_stage1_topk)
        if not retrieved_stage1:
            packaged = self._build_one_shot_guidance(numeric_history, base_text)
            self.cache[cache_key] = packaged
            if len(self.cache) > self.config.cache_size:
                self.cache.popitem(last=False)
            return packaged

        trend_prompt = self._format_trend_prompt(numeric_summary, retrieved_stage1)
        trend_hypothesis = self._generate_trend_hypothesis(
            trend_prompt,
            numeric_summary,
            retrieved_stage1,
            numeric_history,
        )
        trend_hypothesis = self._augment_trend_hypothesis(trend_hypothesis, numeric_stats)
        trend_query_text = self._trend_hypothesis_to_query_text(trend_hypothesis)
        stage2_query = self._build_stage2_query(query, trend_query_text)
        retrieved_stage2 = self._retrieve(stage2_query, top_k=self.rag_stage2_topk)
        if retrieved_stage2:
            final_retrieved = self._merge_retrieved(
                retrieved_stage2,
                retrieved_stage1,
                self.rag_stage2_topk,
            )
        else:
            final_retrieved = retrieved_stage1
        composed_text = self._compose_two_stage_text(
            base_text,
            numeric_summary,
            trend_hypothesis,
            final_retrieved,
        )

        packaged = {
            "cot_text": trend_hypothesis,
            "retrieved_text": " ".join(final_retrieved),
            "composed_text": composed_text,
        }
        self.cache[cache_key] = packaged
        if len(self.cache) > self.config.cache_size:
            self.cache.popitem(last=False)
        return packaged
