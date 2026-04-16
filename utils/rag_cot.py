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
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        device = self._resolve_device(config.device)
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
                model_kwargs["device_map"] = "auto"
            if config.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(
                config.cot_model,
                **model_kwargs,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if device.type == "cuda" and not (config.load_in_8bit or config.load_in_4bit):
                model = model.to(device)
            model.eval()
            return {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
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

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device)
        return torch.device("cpu")

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
            tokenizer = self.generator["tokenizer"]
            model = self.generator["model"]
            device = self.generator["device"]
            token_input = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            )
            if not (self.config.load_in_8bit or self.config.load_in_4bit):
                token_input = {k: v.to(device) for k, v in token_input.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **token_input,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                    num_return_sequences=1,
                    pad_token_id=self.generator["pad_token_id"],
                    eos_token_id=self.generator["eos_token_id"],
                )
            generated_ids = output_ids[0][token_input["input_ids"].shape[1]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return text.strip()
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

    def _compose_text(self, base_text: str, retrieved: List[str], cot_text: str) -> str:
        blocks = []
        if base_text and base_text != "NA":
            blocks.append(base_text)
        if retrieved:
            blocks.append("Retrieved evidence: " + " ".join(retrieved))
        if cot_text and self.config.include_cot_in_text:
            blocks.append("Intermediate trend reasoning: " + cot_text)
        return "\n".join(blocks) if blocks else ""

    def _compose_two_stage_text(
        self,
        base_text: str,
        numeric_summary: str,
        trend_hypothesis: str,
        retrieved: List[str],
    ) -> str:
        raw_text = base_text if not self._is_empty_text(base_text) else "NA"
        lines = [
            "[RAW TEXT]",
            raw_text,
            "",
            "[NUMERICAL SUMMARY]",
            numeric_summary,
            "",
            "[TREND HYPOTHESIS]",
            trend_hypothesis,
            "",
            "[RETRIEVED EVIDENCE - REFINED]",
        ]
        if retrieved:
            for idx, item in enumerate(retrieved, start=1):
                lines.append(f"{idx}) {self._truncate_evidence(item)}")
        else:
            lines.append("1) NA")
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
        composed_text = self._compose_text(base_text, retrieved, cot_text)
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
