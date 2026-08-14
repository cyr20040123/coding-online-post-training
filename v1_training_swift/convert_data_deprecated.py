#!/usr/bin/env python3
"""
将 verl 用的 Parquet 训练数据转换为 ms-swift 兼容的 JSONL 格式。

ms-swift 支持 messages (OpenAI chat format) JSONL，与当前 Parquet 格式相同。
"""
import argparse
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert verl Parquet to swift JSONL")
    parser.add_argument("--input", "-i", required=True, help="Input Parquet file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            msgs = row["messages"]
            # Convert numpy array to list if needed
            if hasattr(msgs, "tolist"):
                msgs = msgs.tolist()
            sample = {"messages": msgs}
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Converted {len(df)} samples to {args.output}")


if __name__ == "__main__":
    main()
