import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings
from utils.timefeatures import time_features
from utils.rag_cot import RAGCoTConfig, RAGCoTPipeline
from utils.trend_prior import build_trend_fields, trend_fields_to_vector
from utils.prepare4llm import get_desc

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
except ImportError:
    class StandardScaler:  # minimal fallback
        def fit(self, X):
            self.mean_ = np.mean(X, axis=0, keepdims=True)
            self.scale_ = np.std(X, axis=0, ddof=0, keepdims=True)
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X):
            return (X - self.mean_) / self.scale_

        def inverse_transform(self, X):
            return X * self.scale_ + self.mean_

    class MinMaxScaler:  # minimal fallback, feature_range=(-1, 1)
        def __init__(self, feature_range=(-1, 1)):
            self.feature_range = feature_range

        def fit(self, X):
            self.data_min_ = np.min(X, axis=0, keepdims=True)
            self.data_max_ = np.max(X, axis=0, keepdims=True)
            self.scale_ = self.data_max_ - self.data_min_
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X):
            min_a, max_a = self.feature_range
            norm = (X - self.data_min_) / self.scale_
            return norm * (max_a - min_a) + min_a

        def inverse_transform(self, X):
            min_a, max_a = self.feature_range
            norm = (X - min_a) / (max_a - min_a)
            return norm * self.scale_ + self.data_min_

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 text_len=1, scaler_type='standard',
                 max_text_tokens=256, text_drop_prob=0.0,
                 use_rag_cot=False, rag_topk=3, cot_model=None,
                 cot_max_new_tokens=96, cot_temperature=0.7,
                 cot_cache_size=1024, cot_device=None, rag_use_retrieval=True,
                 cot_load_in_8bit=False, cot_load_in_4bit=False, trend_cfg=False,
                 use_two_stage_rag=False, rag_stage1_topk=-1, rag_stage2_topk=-1,
                 two_stage_gate=True, trend_slope_eps=1e-3,
                 rag_consistency=False, consistency_unknown_penalty=1.0,
                 consistency_conflict_penalty=0.5,
                 use_scale_router=False, scale_route_horizons=None,
                 scale_window_candidates=None, scale_route_temperature=0.20,
                 use_text_control_router=False, text_control_mix=0.35,
                 text_control_bin_weights=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        # init
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.text_len = text_len
        self.max_text_tokens = max_text_tokens
        self.text_drop_prob = text_drop_prob
        self.use_rag_cot = use_rag_cot
        self.rag_topk = rag_topk
        self.cot_model = cot_model
        self.cot_max_new_tokens = cot_max_new_tokens
        self.cot_temperature = cot_temperature
        self.cot_cache_size = cot_cache_size
        self.cot_device = cot_device
        self.cot_load_in_8bit = cot_load_in_8bit
        self.cot_load_in_4bit = cot_load_in_4bit
        self.rag_use_retrieval = rag_use_retrieval
        self.trend_cfg = trend_cfg
        self.use_two_stage_rag = use_two_stage_rag
        self.rag_stage1_topk = rag_stage1_topk
        self.rag_stage2_topk = rag_stage2_topk
        self.two_stage_gate = two_stage_gate
        self.trend_slope_eps = trend_slope_eps
        self.rag_consistency = rag_consistency
        self.consistency_unknown_penalty = consistency_unknown_penalty
        self.consistency_conflict_penalty = consistency_conflict_penalty
        self.use_scale_router = use_scale_router
        self.scale_route_temperature = max(float(scale_route_temperature), 1e-3)
        self.use_text_control_router = bool(use_text_control_router)
        self.text_control_mix = float(np.clip(text_control_mix, 0.0, 1.0))
        self._text_control_bin_weights_cfg = text_control_bin_weights
        self.guidance_cache = {}

        self.root_path = root_path
        self.data_path = data_path
        self.data_prefix = data_path.split('.')[0]
        
        if scale:
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                scaler_type = 'minmax'
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        self.scaler_type = scaler_type
        self.__read_data__()
        self.domain = data_path.split('/')[0]
        self.desc = get_desc(self.domain, self.seq_len, self.pred_len)
        self.scale_route_horizons = self._resolve_scale_route_horizons(
            scale_route_horizons,
            scale_window_candidates,
        )
        self.scale_num_bins = len(self.scale_route_horizons)
        self.scale_window_candidates = self._resolve_scale_window_candidates(
            scale_window_candidates,
            self.scale_num_bins,
        )
        self.text_control_bin_weights = self._resolve_text_control_bin_weights(
            self._text_control_bin_weights_cfg
        )
        if self.use_scale_router and self.scale_num_bins == 0:
            warnings.warn(
                "Scale router was enabled but no valid bins could be resolved; falling back to fixed text windows."
            )
            self.use_scale_router = False
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.use_rag_cot:
            rag_cfg = RAGCoTConfig(
                top_k=self.rag_topk,
                max_new_tokens=self.cot_max_new_tokens,
                temperature=self.cot_temperature,
                cot_model=self.cot_model,
                cache_size=self.cot_cache_size,
                device=self.cot_device,
                use_retrieval=self.rag_use_retrieval,
                trust_remote_code=True if self.cot_model and ("qwen" in self.cot_model.lower()) else False,
                load_in_8bit=self.cot_load_in_8bit,
                load_in_4bit=self.cot_load_in_4bit,
                structured_output=self.trend_cfg,
                include_cot_in_text=not self.trend_cfg,
                use_two_stage_rag=self.use_two_stage_rag,
                rag_stage1_topk=self.rag_stage1_topk,
                rag_stage2_topk=self.rag_stage2_topk,
                two_stage_gate=self.two_stage_gate,
                trend_slope_eps=self.trend_slope_eps,
                rag_consistency=self.rag_consistency,
                consistency_unknown_penalty=self.consistency_unknown_penalty,
                consistency_conflict_penalty=self.consistency_conflict_penalty,
            )
            self.rag_cot = RAGCoTPipeline(
                domain=self.domain,
                search_df=self.search_df,
                desc=self.desc,
                lookback_len=self.seq_len,
                pred_len=self.pred_len,
                config=rag_cfg,
            )
        else:
            self.rag_cot = None
        

    def __read_data__(self):
        df_num = pd.read_csv(os.path.join(self.root_path, 'numerical', self.data_path))
        df_report = pd.read_csv(os.path.join(self.root_path, 'textual', self.data_prefix + '_report.csv'))
        df_search = pd.read_csv(os.path.join(self.root_path, 'textual', self.data_prefix + '_search.csv'))

        df_num = df_num.dropna(axis='index', how='any', subset=['OT'])
        df_report = df_report.dropna(axis='index', how='any', subset=['fact'])
        df_search = df_search.dropna(axis='index', how='any', subset=['fact'])

        df_num['date'], df_num['start_date'], df_num['end_date'] = pd.to_datetime(df_num['date']), pd.to_datetime(df_num['start_date']), pd.to_datetime(df_num['end_date'])
        df_report['start_date'], df_report['end_date'] = pd.to_datetime(df_report['start_date']), pd.to_datetime(df_report['end_date'])
        df_search['start_date'], df_search['end_date'] = pd.to_datetime(df_search['start_date']), pd.to_datetime(df_search['end_date'])

        df_num = df_num.sort_values('date', ascending=True).reset_index(drop=True)
        df_report = df_report.sort_values('start_date', ascending=True).reset_index(drop=True)
        df_search = df_search.sort_values('start_date', ascending=True).reset_index(drop=True)
        num_train = int(len(df_num) * 0.7)
        num_test = int(len(df_num) * 0.2)
        num_vali = len(df_num) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_num) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_num)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        
        first_start_date = df_num.start_date[border1]
        final_end_date = df_num.end_date[border2-1]

        df_data = df_num[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
            self.mean_data = self.scaler.mean_
            self.std_data = self.scaler.scale_
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_num[['date']][border1:border2]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['is_weekend'] = df_stamp.weekday.apply(lambda row: 1 if row >= 5 else 0, 1)
            df_stamp['is_month_end'] = df_stamp.date.apply(lambda row: 1 if row.is_month_end else 0, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            weekend_flag = (df_stamp.date.dt.weekday >= 5).astype(np.float32).values.reshape(-1, 1)
            month_end_flag = df_stamp.date.dt.is_month_end.astype(np.float32).values.reshape(-1, 1)
            data_stamp = np.concatenate([data_stamp, weekend_flag, month_end_flag], axis=1)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


        self.data_stamp = data_stamp
        self.num_dates = df_num[['start_date', 'end_date']][border1:border2].reset_index(drop=True)
        self.txt_report = df_report[['start_date', 'end_date', 'fact']].loc[(df_report.end_date >= first_start_date) & (df_report.end_date <= final_end_date)]
        self.search_df = df_search[['start_date', 'end_date', 'fact']]

    def _coerce_positive_ints(self, values):
        if values is None:
            return []
        if isinstance(values, str):
            values = [item.strip() for item in values.split(',')]
        parsed = []
        for item in values:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value > 0:
                parsed.append(value)
        return parsed

    def _auto_scale_horizons(self, num_bins):
        if num_bins <= 0 or self.pred_len <= 0:
            return []
        steps = np.arange(1, self.pred_len + 1)
        splits = [chunk for chunk in np.array_split(steps, num_bins) if len(chunk) > 0]
        return [int(chunk[-1]) for chunk in splits]

    def _resolve_scale_route_horizons(self, scale_route_horizons, scale_window_candidates):
        horizons = sorted(set(self._coerce_positive_ints(scale_route_horizons)))
        if len(horizons) > 0:
            return horizons
        candidate_windows = self._coerce_positive_ints(scale_window_candidates)
        if len(candidate_windows) > 0:
            return self._auto_scale_horizons(len(candidate_windows))
        if not self.use_scale_router:
            return []
        return self._auto_scale_horizons(4)

    def _resolve_scale_window_candidates(self, scale_window_candidates, num_bins):
        if num_bins <= 0:
            return []
        explicit = self._coerce_positive_ints(scale_window_candidates)
        if explicit:
            clipped = [max(1, min(int(v), self.text_len)) for v in explicit]
            if len(clipped) == num_bins:
                return clipped
            warnings.warn(
                f"scale_window_candidates expects {num_bins} values, got {len(clipped)}; using evenly spaced defaults instead."
            )
        max_window = max(int(self.text_len), 1)
        return [
            max(1, min(max_window, int(round(((idx + 1) * max_window) / num_bins))))
            for idx in range(num_bins)
        ]

    def _prepare_scale_series(self, numeric_history):
        arr = np.asarray(numeric_history, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim == 1:
            return arr
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr.mean(axis=1)

    def _safe_autocorr(self, arr, lag):
        if lag <= 0 or arr.size <= lag:
            return 0.0
        left = arr[:-lag]
        right = arr[lag:]
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        denom = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
        if denom <= 1e-8:
            return 1.0
        return float(np.dot(left_centered, right_centered) / denom)

    def _estimate_period_ratio(self, arr):
        if arr.size < 4:
            return 0.5
        trend_line = np.linspace(arr[0], arr[-1], arr.size)
        detrended = arr - trend_line
        spectrum = np.abs(np.fft.rfft(detrended))[1:]
        if spectrum.size == 0 or np.max(spectrum) <= 1e-8:
            return 0.5
        dominant_idx = int(np.argmax(spectrum)) + 1
        dominant_period = float(arr.size) / float(dominant_idx)
        return float(np.clip(dominant_period / max(float(arr.size), 1.0), 0.0, 1.0))

    def _softmax_np(self, logits):
        logits = np.asarray(logits, dtype=float)
        if logits.size == 0:
            return np.zeros((0,), dtype=np.float32)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        denom = np.sum(probs)
        if denom <= 0:
            return np.full(logits.shape, 1.0 / logits.size, dtype=np.float32)
        return (probs / denom).astype(np.float32)

    def _compute_scale_route(self, numeric_history):
        if not self.use_scale_router or self.scale_num_bins <= 0:
            return np.zeros((0,), dtype=np.float32)
        if self.scale_num_bins == 1:
            return np.ones((1,), dtype=np.float32)

        arr = self._prepare_scale_series(numeric_history)
        if arr.size < 2:
            return np.full((self.scale_num_bins,), 1.0 / self.scale_num_bins, dtype=np.float32)

        slope = abs(float(arr[-1] - arr[0])) / max(arr.size - 1, 1)
        std = float(np.std(arr))
        local_vol = float(np.std(np.diff(arr))) if arr.size > 2 else 0.0
        trend_strength = slope / (slope + local_vol + 1e-6)
        lag = max(1, arr.size // 3)
        persistence = 0.5 * (self._safe_autocorr(arr, lag) + 1.0)
        periodicity = self._estimate_period_ratio(arr)
        stability = 1.0 - (local_vol / (local_vol + std + 1e-6))

        longness = np.clip(
            0.35 * trend_strength +
            0.30 * persistence +
            0.20 * periodicity +
            0.15 * stability,
            0.0,
            1.0,
        )
        positions = np.linspace(0.0, 1.0, self.scale_num_bins)
        logits = -((longness - positions) ** 2) / (2.0 * (self.scale_route_temperature ** 2))
        return self._softmax_np(logits)

    def _select_text_window(self, scale_route):
        if (not self.use_scale_router) or len(self.scale_window_candidates) == 0:
            return int(self.text_len)
        weights = np.asarray(scale_route, dtype=float)
        if weights.size != len(self.scale_window_candidates):
            weights = np.full((len(self.scale_window_candidates),), 1.0 / len(self.scale_window_candidates))
        candidates = np.asarray(self.scale_window_candidates, dtype=float)
        window = int(round(float(np.dot(weights, candidates))))
        return max(1, min(int(self.text_len), window))

    def _resolve_text_control_bin_weights(self, values):
        if self.scale_num_bins <= 0:
            return np.zeros((0,), dtype=np.float32)
        if values is None:
            values = []
        weights = []
        for item in values:
            try:
                weights.append(float(item))
            except (TypeError, ValueError):
                continue
        if len(weights) == 0:
            return np.ones((self.scale_num_bins,), dtype=np.float32)
        arr = np.asarray(weights, dtype=np.float32)
        if arr.size != self.scale_num_bins:
            x_old = np.linspace(0.0, 1.0, arr.size)
            x_new = np.linspace(0.0, 1.0, self.scale_num_bins)
            arr = np.interp(x_new, x_old, arr).astype(np.float32)
        return np.clip(arr, 0.0, 1.0)

    def _compute_text_control_route(self, trend_prior):
        if self.scale_num_bins <= 0:
            return np.zeros((0,), dtype=np.float32)
        if self.scale_num_bins == 1:
            return np.ones((1,), dtype=np.float32)
        trend_vec = np.asarray(trend_prior, dtype=np.float32).reshape(-1)
        if trend_vec.size < 3:
            return np.full((self.scale_num_bins,), 1.0 / self.scale_num_bins, dtype=np.float32)
        direction = float(np.clip(abs(trend_vec[0]), 0.0, 1.0))
        strength = float(np.clip((trend_vec[1] - 0.5) / 1.0, 0.0, 1.0))
        stability = float(1.0 - np.clip(trend_vec[2], 0.0, 1.0))
        longness = np.clip(
            0.45 * strength +
            0.35 * stability +
            0.20 * direction,
            0.0,
            1.0,
        )
        positions = np.linspace(0.0, 1.0, self.scale_num_bins)
        logits = -((longness - positions) ** 2) / (2.0 * (self.scale_route_temperature ** 2))
        return self._softmax_np(logits)

    def _blend_text_control_route(self, base_scale_route, trend_prior, text_mark, consistency_score):
        base = np.asarray(base_scale_route, dtype=np.float32)
        if (not self.use_text_control_router) or base.size == 0 or int(text_mark) <= 0:
            return base
        text_route = self._compute_text_control_route(trend_prior)
        consistency = float(np.clip(consistency_score, 0.0, 1.0))
        trend_vec = np.asarray(trend_prior, dtype=np.float32).reshape(-1)
        if trend_vec.size < 3:
            return base
        strength = float(np.clip((trend_vec[1] - 0.5) / 1.0, 0.0, 1.0))
        stability = float(1.0 - np.clip(trend_vec[2], 0.0, 1.0))
        mix = np.clip(
            self.text_control_mix *
            consistency *
            (0.5 + 0.5 * strength) *
            (0.5 + 0.5 * stability),
            0.0,
            1.0,
        )
        bin_mix = mix * self.text_control_bin_weights
        fused = (1.0 - bin_mix) * base + bin_mix * text_route
        denom = float(np.sum(fused))
        if denom <= 0:
            return base
        return (fused / denom).astype(np.float32)

    def collect_text(self, start_date, end_date):
        report = self.txt_report.loc[(self.txt_report.end_date >= start_date) & (self.txt_report.end_date <= end_date)]
        def add_datemark(row):
            return row['start_date'].strftime("%Y-%m-%d") + " to " + row['end_date'].strftime("%Y-%m-%d") + ": " + row['fact']
        if not report.empty:
            report = report.apply(add_datemark, axis=1).to_list()
            report.insert(0, self.desc)
            text_mark = 1
        else:
            report = ['NA']
            text_mark = 0
        # deduplicate while keeping order
        seen = set()
        filtered_report = []
        for segment in report:
            if segment not in seen:
                filtered_report.append(segment)
                seen.add(segment)
        all_txt = ' '.join(filtered_report)
        # basic cleanup and truncation for robustness
        tokens = all_txt.split()
        if len(tokens) > self.max_text_tokens:
            tokens = tokens[:self.max_text_tokens]
        all_txt = ' '.join(tokens)
        return all_txt, text_mark
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
    
        seq_x_stamp = self.data_stamp[s_begin:s_end]
        seq_y_stamp = self.data_stamp[r_begin:r_end]

        base_scale_route = self._compute_scale_route(seq_x)
        dynamic_text_len = self._select_text_window(base_scale_route)

        text_begin = max(s_end - dynamic_text_len, 0)
        text_end = s_end

        seq_x_txt, txt_mark = self.collect_text(self.num_dates.start_date[text_begin], self.num_dates.end_date[text_end])
        text_dropped = False
        if (self.text_drop_prob > 0) and (np.random.rand() < self.text_drop_prob):
            seq_x_txt, txt_mark = 'NA', 0
            text_dropped = True
        rag_retrieved, cot_text = '', ''
        consistency_score = 1.0
        if self.use_rag_cot and self.rag_cot is not None and not text_dropped:
            cached = self.guidance_cache.get(index, None)
            if cached is None:
                guidance = self.rag_cot.build_guidance_text(
                    numeric_history=seq_x,
                    start_date=self.num_dates.start_date[text_begin],
                    end_date=self.num_dates.end_date[text_end - 1],
                    base_text=seq_x_txt,
                )
                seq_x_txt = guidance["composed_text"]
                cot_text = guidance["cot_text"]
                rag_retrieved = guidance["retrieved_text"]
                consistency_score = float(guidance.get("consistency_score", 1.0))
                txt_mark = 1 if len(seq_x_txt.strip()) > 0 else 0
                self.guidance_cache[index] = (seq_x_txt, txt_mark, cot_text, rag_retrieved, consistency_score)
            else:
                seq_x_txt, txt_mark, cot_text, rag_retrieved, consistency_score = cached
        if len(seq_x_txt.strip()) == 0 or seq_x_txt == 'NA':
            txt_mark = 0

        trend_fields = build_trend_fields(cot_text, seq_x)
        trend_prior = trend_fields_to_vector(trend_fields)
        scale_route = self._blend_text_control_route(base_scale_route, trend_prior, txt_mark, consistency_score)

        observed_data = np.concatenate([seq_x, seq_y], axis=0)
        timesteps = np.concatenate([seq_x_stamp, seq_y_stamp], axis=0)
        observed_mask = np.ones_like(observed_data)
        gt_mask = np.concatenate([np.ones_like(seq_x), np.zeros_like(seq_y)], axis=0)

        s = {
            'observed_data': observed_data,
            'observed_mask': observed_mask,
            'gt_mask': gt_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len).astype(np.float32), 
            'feature_id': np.arange(seq_x.shape[1]).astype(np.float32),
            'timesteps': timesteps,
            'texts': seq_x_txt,
            'text_mark': txt_mark,
            'cot_text': cot_text,
            'retrieved_text': rag_retrieved,
            'trend_prior': trend_prior,
            'scale_route': scale_route,
            'base_scale_route': base_scale_route,
            'text_control_bin_weights': self.text_control_bin_weights.astype(np.float32),
            'scale_window': np.int64(dynamic_text_len),
            'consistency_score': np.float32(consistency_score),
        }

        return s

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', target='OT',
                 scale=True, timeenc=0, freq='h', text_len=None,
                 scaler_type='minmax'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        if scale:
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                scaler_type = 'minmax'
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        self.scaler_type = scaler_type

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv', target='OT',
                 scale=True, timeenc=0, freq='t', text_len=None,
                 scaler_type='minmax'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        if scale:
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                scaler_type = 'minmax'
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        self.scaler_type = scaler_type
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        observed_data = np.concatenate(seq_x, seq_y, axis=0)
        timesteps = np.concatenate(seq_x_mark, seq_y_mark, axis=0)
        observed_mask = np.ones_like(observed_data)
        gt_mask = np.concatenate(np.ones_like(seq_x), np.zeros_like(seq_y), axis=0)

        s = {
            'observed_data': observed_data,
            'observed_mask': observed_mask,
            'gt_mask': gt_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len) * 1.0, 
            'feature_id': np.arange(self.seq_x.shape[1]) * 1.0,
            'timesteps': timesteps

        }

        return s

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
