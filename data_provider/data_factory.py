from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
import torch
from torch.utils.data import DataLoader

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
        extra_kwargs['aug_noise_std'] = getattr(args, "aug_noise_std", 0.0)
        extra_kwargs['aug_time_warp_prob'] = getattr(args, "aug_time_warp_prob", 0.0)
        extra_kwargs['aug_segment_scale_std'] = getattr(args, "aug_segment_scale_std", 0.1)
        extra_kwargs['adaptive_noise_scale'] = getattr(args, "adaptive_noise_scale", 0.0)
        extra_kwargs['text_quality_gate'] = getattr(args, "text_quality_gate", True)
        extra_kwargs['text_quality_min_scale'] = getattr(args, "text_quality_min_scale", 0.0)
        extra_kwargs['text_quality_coverage_mix'] = getattr(args, "text_quality_coverage_mix", 0.5)
        extra_kwargs['text_recency_tau_days'] = getattr(args, "text_recency_tau_days", 14.0)
        extra_kwargs['text_coverage_kappa'] = getattr(args, "text_coverage_kappa", 3.0)
        extra_kwargs['text_quality_weights'] = getattr(args, "text_quality_weights", [0.5, 0.3, 0.2])
        extra_kwargs['text_trust_ret'] = getattr(args, "text_trust_ret", 0.75)
        extra_kwargs['text_trust_cot'] = getattr(args, "text_trust_cot", 0.5)
        extra_kwargs['text_quality_drop_threshold'] = getattr(args, "text_quality_drop_threshold", 0.3)
        extra_kwargs['text_quality_mid_threshold'] = getattr(args, "text_quality_mid_threshold", 0.6)
        extra_kwargs['text_trend_ret_scale'] = getattr(args, "text_trend_ret_scale", 0.5)
        extra_kwargs['text_trend_cot_scale'] = getattr(args, "text_trend_cot_scale", 0.3)
        extra_kwargs['text_trend_raw_weight'] = getattr(args, "text_trend_raw_weight", 1.0)
        extra_kwargs['text_trend_ret_weight'] = getattr(args, "text_trend_ret_weight", 0.35)
        extra_kwargs['text_trend_cot_weight'] = getattr(args, "text_trend_cot_weight", 0.15)
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
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
