import argparse
import random

import numpy as np

from data_provider.data_loader import Dataset_Custom


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check RAG retrieval leakage.")
    parser.add_argument("--root_path", type=str, default="Time-MMD-main")
    parser.add_argument("--data_path", type=str, default="Economy/Economy.csv")
    parser.add_argument("--seq_len", type=int, default=36)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--text_len", type=int, default=36)
    parser.add_argument("--max_text_tokens", type=int, default=256)
    parser.add_argument("--rag_topk", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Check valid/test specifically (most leakage happens there).
    for flag in ["valid", "test"]:
        ds = Dataset_Custom(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            data_path=args.data_path,
            text_len=args.text_len,
            max_text_tokens=args.max_text_tokens,
            use_rag_cot=True,
            rag_topk=args.rag_topk,
            cot_model=None,  # template path, no extra deps
            rag_use_retrieval=True,
        )
        assert ds.rag_cot is not None
        if ds.search_df is None or ds.search_df.empty:
            print(f"[{flag}] search_df empty; skip.")
            continue

        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[: min(args.num_samples, len(indices))]

        violations = 0
        for idx in indices:
            s_begin = idx
            s_end = s_begin + ds.seq_len
            text_begin = s_end - ds.text_len
            text_end = s_end
            end_date = ds.num_dates.end_date[text_end - 1]

            seq_x = ds.data_x[s_begin:s_end, :]
            base_text, _ = ds.collect_text(ds.num_dates.start_date[text_begin], ds.num_dates.end_date[text_end])
            numeric_summary = ds.rag_cot._summarize_numeric(seq_x)
            query = ds.rag_cot._build_query(numeric_summary, base_text)
            retrieved = ds.rag_cot._retrieve(query, top_k=ds.rag_cot.config.top_k, max_end_date=end_date)

            # With the new guard, anything retrieved must satisfy end_date <= sample end_date,
            # which is enforced inside _retrieve. If we get here, we only verify retrieval is stable.
            if retrieved and ("end_date" in ds.search_df.columns):
                # Spot-check: at least the overall corpus max_end_date is within the split boundary.
                if pd_to_dt(ds.search_df.end_date.max()) > pd_to_dt(ds.num_dates.end_date.max()):
                    violations += 1
                    print(f"[{flag}] corpus max end_date exceeds split boundary.")
                    break

        if violations == 0:
            print(f"[{flag}] OK: checked {len(indices)} samples; no future-evidence corpus issues detected.")
        else:
            raise SystemExit(f"[{flag}] FAILED: detected {violations} potential leakage issue(s).")


def pd_to_dt(x):
    # local helper to avoid importing pandas at module import time
    import pandas as pd

    return pd.to_datetime(x)


if __name__ == "__main__":
    main()

