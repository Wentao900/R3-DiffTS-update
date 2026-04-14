from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

import numpy as np
from utils.trend_prior import build_trend_fields, trend_fields_to_vector

data_dict = {
    'custom': Dataset_Custom,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    drop_last = False if flag == 'test' else True
    batch_size = args.batch_size
    freq = args.freq
    extra_kwargs = {'text_len': args.text_len}
    if args.data == 'custom':
        extra_kwargs['max_text_tokens'] = args.max_text_tokens
        extra_kwargs['text_drop_prob'] = args.text_drop_prob
        extra_kwargs['use_rag_cot'] = args.use_rag_cot
        extra_kwargs['rag_topk'] = args.rag_topk
        extra_kwargs['cot_model'] = args.cot_model
        extra_kwargs['cot_max_new_tokens'] = args.cot_max_new_tokens
        extra_kwargs['cot_temperature'] = args.cot_temperature
        extra_kwargs['cot_cache_size'] = args.cot_cache_size
        extra_kwargs['cot_device'] = args.cot_device
        extra_kwargs['cot_load_in_8bit'] = getattr(args, "cot_load_in_8bit", False)
        extra_kwargs['cot_load_in_4bit'] = getattr(args, "cot_load_in_4bit", False)
        extra_kwargs['rag_use_retrieval'] = not args.cot_only
        extra_kwargs['trend_cfg'] = getattr(args, "trend_cfg", False)
        extra_kwargs['use_two_stage_rag'] = getattr(args, "use_two_stage_rag", False)
        extra_kwargs['rag_stage1_topk'] = getattr(args, "rag_stage1_topk", -1)
        extra_kwargs['rag_stage2_topk'] = getattr(args, "rag_stage2_topk", -1)
        extra_kwargs['two_stage_gate'] = getattr(args, "two_stage_gate", True)
        extra_kwargs['trend_slope_eps'] = getattr(args, "trend_slope_eps", 1e-3)
        # Pass LLM name so dataset can apply tokenizer-aware truncation.
        extra_kwargs['llm'] = getattr(args, "llm", None)
        # When CoT runs on GPU, batching in collate is faster and avoids GPU work in workers.
        extra_kwargs['rag_cot_in_collate'] = True
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        **extra_kwargs
    )
    print(flag, len(data_set))

    collate_fn = None
    num_workers = args.num_workers
    # Enable batched CoT generation in collate when using a real CoT model.
    if getattr(data_set, "use_rag_cot", False) and getattr(data_set, "rag_cot", None) is not None:
        cot_model = getattr(data_set, "cot_model", None)
        has_generator = getattr(data_set.rag_cot, "_cot_model", None) is not None
        if cot_model and has_generator and getattr(data_set, "rag_cot_in_collate", False):
            # Force collate to run in the main process (workers cannot share GPU model).
            num_workers = 0

            def collate_fn(batch, dataset=data_set):
                need_idx = [i for i, s in enumerate(batch) if int(s.get("rag_need_guidance", 0)) == 1]
                if need_idx:
                    numeric_histories = [batch[i]["rag_numeric_history"] for i in need_idx]
                    start_dates = [batch[i]["rag_start_date"] for i in need_idx]
                    end_dates = [batch[i]["rag_end_date"] for i in need_idx]
                    base_texts = [batch[i]["rag_base_text"] for i in need_idx]
                    guidances = dataset.rag_cot.build_guidance_text_batch(
                        numeric_histories=numeric_histories,
                        start_dates=start_dates,
                        end_dates=end_dates,
                        base_texts=base_texts,
                    )
                    for i, g in zip(need_idx, guidances):
                        composed = g.get("composed_text", "")
                        cot_text = g.get("cot_text", "")
                        retrieved_text = g.get("retrieved_text", "")
                        composed = dataset._truncate_final_text(composed)
                        batch[i]["texts"] = composed
                        batch[i]["text_mark"] = 1 if len(str(composed).strip()) > 0 else 0
                        batch[i]["cot_text"] = cot_text
                        batch[i]["retrieved_text"] = retrieved_text
                        # Refresh trend prior based on the final CoT/trend text
                        seq_x = batch[i]["rag_numeric_history"]
                        trend_fields = build_trend_fields(cot_text, seq_x)
                        batch[i]["trend_prior"] = trend_fields_to_vector(trend_fields)
                        batch[i]["text_reliability"] = np.float32(
                            dataset._estimate_text_reliability(seq_x, batch[i]["text_mark"], cot_text)
                        )
                # Sanitize Nones to keep default_collate happy
                for s in batch:
                    for k in ("texts", "cot_text", "retrieved_text"):
                        if k in s and s[k] is None:
                            s[k] = ""
                    if "trend_prior" in s and s["trend_prior"] is None:
                        s["trend_prior"] = trend_fields_to_vector({"direction": "flat", "strength": "moderate", "volatility": "medium"})
                    if "text_reliability" in s and s["text_reliability"] is None:
                        s["text_reliability"] = np.float32(0.0)
                    # any other None -> empty string (rare safety)
                    for k, v in list(s.items()):
                        if v is None:
                            s[k] = ""
                # Remove rag-only fields to keep batch clean
                for s in batch:
                    for k in ("rag_base_text", "rag_start_date", "rag_end_date", "rag_numeric_history", "rag_need_guidance"):
                        if k in s:
                            s.pop(k)
                return default_collate(batch)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn if collate_fn is not None else None,
    )
    return data_set, data_loader
