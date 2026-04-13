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
                 two_stage_gate=True, trend_slope_eps=1e-3):
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
        
    def _truncate_final_text(self, text: str) -> str:
        """
        Truncate the FINAL composed text to max_text_tokens.
        This prevents downstream tokenizer-side truncation from silently dropping
        the most relevant (recent) evidence/CoT appended by RAG.
        Strategy:
        - If text begins with self.desc: keep the desc prefix, then keep the tail
          of the remaining tokens to fit the budget.
        - Otherwise: keep the tail tokens.
        """
        if not text or text == "NA":
            return text
        if not self.max_text_tokens or int(self.max_text_tokens) <= 0:
            return text
        max_tok = int(self.max_text_tokens)
        tokens = str(text).split()
        if len(tokens) <= max_tok:
            return text
        desc_tokens = str(self.desc).split() if getattr(self, "desc", None) else []
        if desc_tokens and text.strip().startswith(str(self.desc).strip()):
            if len(desc_tokens) >= max_tok:
                return " ".join(desc_tokens[-max_tok:])
            remaining_budget = max_tok - len(desc_tokens)
            rest_tokens = tokens[len(desc_tokens) :]
            kept = desc_tokens + rest_tokens[-remaining_budget:]
            return " ".join(kept)
        return " ".join(tokens[-max_tok:])

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
        # RAG leakage guard:
        # retrieval corpus must not include "future" evidence beyond the current split boundary.
        # Otherwise valid/test will accidentally retrieve facts from later periods (data leakage).
        df_search_cut = df_search[['start_date', 'end_date', 'fact']].copy()
        df_search_cut = df_search_cut.loc[df_search_cut.end_date <= final_end_date]
        if not df_search_cut.empty:
            max_end = df_search_cut.end_date.max()
            if pd.notna(max_end):
                assert max_end <= final_end_date, "search corpus contains future evidence beyond split boundary"
        self.search_df = df_search_cut.reset_index(drop=True)

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
            # Keep the most recent tokens (tail) so the text budget prioritizes
            # segments closest to the forecast window end.
            tokens = tokens[-self.max_text_tokens:]
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

        text_begin = s_end - self.text_len
        text_end = s_end

        seq_x_txt, txt_mark = self.collect_text(self.num_dates.start_date[text_begin], self.num_dates.end_date[text_end])
        text_dropped = False
        if (self.text_drop_prob > 0) and (np.random.rand() < self.text_drop_prob):
            seq_x_txt, txt_mark = 'NA', 0
            text_dropped = True
        rag_retrieved, cot_text = '', ''
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
                seq_x_txt = self._truncate_final_text(seq_x_txt)
                txt_mark = 1 if len(seq_x_txt.strip()) > 0 else 0
                self.guidance_cache[index] = (seq_x_txt, txt_mark, cot_text, rag_retrieved)
            else:
                seq_x_txt, txt_mark, cot_text, rag_retrieved = cached
        if len(seq_x_txt.strip()) == 0 or seq_x_txt == 'NA':
            txt_mark = 0

        trend_fields = build_trend_fields(cot_text, seq_x)
        trend_prior = trend_fields_to_vector(trend_fields)

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
            'trend_prior': trend_prior
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
