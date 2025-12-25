import argparse
import random

import numpy as np

from data_provider.data_loader import Dataset_Custom


def _format_list(items, title):
    print(title)
    if not items:
        print("  (empty)")
        return
    for idx, item in enumerate(items, start=1):
        print(f"  {idx}) {item}")


def main():
    parser = argparse.ArgumentParser(description="Quick sanity check for two-stage RAG.")
    parser.add_argument("--root_path", type=str, default="Time-MMD-main")
    parser.add_argument("--data_path", type=str, default="Economy/Economy.csv")
    parser.add_argument("--flag", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--seq_len", type=int, default=36)
    parser.add_argument("--pred_len", type=int, default=18)
    parser.add_argument("--text_len", type=int, default=36)
    parser.add_argument("--max_text_tokens", type=int, default=256)
    parser.add_argument("--rag_topk", type=int, default=3)
    parser.add_argument("--use_two_stage_rag", action="store_true")
    parser.add_argument("--rag_stage1_topk", type=int, default=-1)
    parser.add_argument("--rag_stage2_topk", type=int, default=-1)
    parser.add_argument("--two_stage_gate", action="store_true", default=True)
    parser.add_argument("--trend_slope_eps", type=float, default=1e-3)
    parser.add_argument("--cot_model", type=str, default=None)
    parser.add_argument("--cot_max_new_tokens", type=int, default=96)
    parser.add_argument("--cot_temperature", type=float, default=0.7)
    parser.add_argument("--cot_cache_size", type=int, default=128)
    parser.add_argument("--cot_device", type=str, default=None)
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = Dataset_Custom(
        root_path=args.root_path,
        flag=args.flag,
        size=[args.seq_len, args.pred_len],
        data_path=args.data_path,
        text_len=args.text_len,
        max_text_tokens=args.max_text_tokens,
        use_rag_cot=True,
        rag_topk=args.rag_topk,
        cot_model=args.cot_model,
        cot_max_new_tokens=args.cot_max_new_tokens,
        cot_temperature=args.cot_temperature,
        cot_cache_size=args.cot_cache_size,
        cot_device=args.cot_device,
        rag_use_retrieval=True,
        trend_cfg=False,
        use_two_stage_rag=args.use_two_stage_rag,
        rag_stage1_topk=args.rag_stage1_topk,
        rag_stage2_topk=args.rag_stage2_topk,
        two_stage_gate=args.two_stage_gate,
        trend_slope_eps=args.trend_slope_eps,
    )

    if dataset.rag_cot is None:
        print("RAG pipeline is disabled; check use_rag_cot and dataset settings.")
        return

    index = args.index if args.index >= 0 else random.randint(0, len(dataset) - 1)
    s_begin = index
    s_end = s_begin + dataset.seq_len
    text_begin = s_end - dataset.text_len
    text_end = s_end

    seq_x = dataset.data_x[s_begin:s_end, :]
    base_text, _ = dataset.collect_text(
        dataset.num_dates.start_date[text_begin],
        dataset.num_dates.end_date[text_end],
    )

    rag = dataset.rag_cot
    numeric_summary = rag._summarize_numeric(seq_x)
    query1 = rag._build_query(numeric_summary, base_text)
    retrieved_stage1 = rag._retrieve(query1, top_k=rag.rag_stage1_topk if args.use_two_stage_rag else None)

    print(f"index={index}")
    print(f"start_date={dataset.num_dates.start_date[text_begin]}")
    print(f"end_date={dataset.num_dates.end_date[text_end - 1]}")
    print("\n[Q1]")
    print(query1)
    _format_list(retrieved_stage1, "\n[E0]")

    trend_hypothesis = ""
    retrieved_stage2 = []
    gate_triggered = False
    if args.use_two_stage_rag:
        numeric_stats = rag._compute_numeric_stats(seq_x)
        gate_triggered = (
            rag.two_stage_gate
            and rag._is_empty_text(base_text)
            and abs(numeric_stats["slope"]) < rag.trend_slope_eps
        )
        if retrieved_stage1:
            trend_prompt = rag._format_trend_prompt(numeric_summary, retrieved_stage1)
            trend_hypothesis = rag._generate_trend_hypothesis(
                trend_prompt,
                numeric_summary,
                retrieved_stage1,
                seq_x,
            )
            query2 = rag._build_stage2_query(query1, trend_hypothesis)
            retrieved_stage2 = rag._retrieve(query2, top_k=rag.rag_stage2_topk)
            print("\n[z0]")
            print(trend_hypothesis)
            print("\n[Q2]")
            print(query2)
            _format_list(retrieved_stage2, "\n[E1]")
        else:
            print("\n[z0]\n(empty; Stage-1 retrieval returned no evidence)")

    guidance = rag.build_guidance_text(
        numeric_history=seq_x,
        start_date=dataset.num_dates.start_date[text_begin],
        end_date=dataset.num_dates.end_date[text_end - 1],
        base_text=base_text,
    )
    final_text = guidance["composed_text"]

    if args.use_two_stage_rag and gate_triggered:
        print("\nGate triggered -> fallback to one-shot.")

    print("\n[FINAL TEXT PREVIEW]")
    preview = final_text[:600].rstrip()
    print(preview + ("..." if len(final_text) > 600 else ""))


if __name__ == "__main__":
    main()
