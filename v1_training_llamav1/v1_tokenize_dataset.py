#!/usr/bin/env python3
"""
Step 1: V1 数据预处理 → token 化 → 保存为 HfDataset

用 LLaMAFactory V1 的 Renderer 做 token 化，产出可直接被 V0 训练配置通过
``tokenized_path`` 加载。

输入: data_openai/ 目录（OpenAI 原生格式，含 reasoning_content + tool_calls）
输出: tokenized_dataset/（HfDataset，含 input_ids / labels / loss_weights）

使用:
    USE_V1=1 python v1_tokenize_dataset.py                  \
        --model Qwen/Qwen3.5-2B                             \
        --data_dir /path/to/data_openai                     \
        --output_dir ./tokenized_dataset                    \
        --cutoff_len 32768                                  \
        --num_workers 8
"""

import argparse
import glob
import json
import os
import sys
from itertools import islice

import torch
from datasets import Dataset as HfDataset
from tqdm import tqdm

# ── Must import converter BEFORE renderer imports ──
import v1_openai_converter  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_json_files(data_dir: str) -> list[dict]:
    """Recursively find and load all JSON files from data_dir."""
    all_items = []
    json_files = sorted(glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True))
    print(f"[Load] Found {len(json_files)} JSON files in {data_dir}")

    for fp in tqdm(json_files, desc="Loading JSON files"):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_items.extend(data)
        elif isinstance(data, dict):
            all_items.append(data)

    print(f"[Load] Total: {len(all_items)} items")
    return all_items


def _prefix_split_samples(converted: list[dict]) -> list[dict]:
    """Split each conversation on every supervised assistant turn."""
    expanded = []
    for sample in converted:
        messages = sample.get("messages", [])
        if not messages:
            continue
        cuts = [i + 1 for i, m in enumerate(messages)
                if m["role"] == "assistant" and m.get("loss_weight", 1.0) > 1e-6]
        if not cuts:
            expanded.append(sample)
        else:
            for cut in cuts:
                expanded.append({**sample, "messages": messages[:cut]})
    return expanded


def _tokenize_one(sample: dict, renderer, cutoff_len: int, loss_norm: str = "token") -> dict | None:
    """Tokenize a single V1 sample. Returns None if all labels masked.

    Args:
        loss_norm: 'token' = 默认行为，监督 token 权重不变（per-token 平均）
                   'turn'  = 监督 span 的 loss_weights 除以 span 长度（per-turn 平均）
                   'sqrt'  = 监督 span 的 loss_weights 除以 sqrt(span 长度)
                             （每轮总权重 ∝ sqrt(len)，per-token 与 per-turn 的折中）
    """
    try:
        model_inputs = renderer.process_samples([sample])
    except Exception:
        return None

    if not model_inputs:
        return None

    mi = model_inputs[0]

    if len(mi["input_ids"]) > cutoff_len:
        # 保留结尾 — V1 的 diff-based labeling 监督最后一个 assistant turn，
        # 它在序列末尾。截断开头（prompt 部分）不影响监督信号。
        for k in ("input_ids", "labels", "loss_weights", "attention_mask", "position_ids"):
            if k in mi:
                mi[k] = mi[k][-cutoff_len:]

    if all(l < 0 for l in mi.get("labels", [])):
        return None

    # ── 轮级归一化：监督 span 权重除以 divisor ──
    # 截断之后再做，保证归一化基于截断后的真实轮长。
    # 每轮总权重 = len × w_token = len / divisor：
    #   'turn'  (divisor=len)      → 每轮总权重 = 1        → 轮间等权
    #   'sqrt'  (divisor=sqrt(len)) → 每轮总权重 = sqrt(len) → 长轮权重更大但被阻尼
    if loss_norm in ("turn", "sqrt"):
        lw = mi.get("loss_weights")
        if lw is not None:
            # loss_weights 可能是 tensor 或 list，统一处理
            if isinstance(lw, torch.Tensor):
                span = lw > 0
                span_len = int(span.sum().item())
                if span_len > 0:
                    divisor = span_len if loss_norm == "turn" else span_len ** 0.5
                    mi["loss_weights"] = lw / divisor
            else:
                span_len = sum(1 for w in lw if w > 0)
                if span_len > 0:
                    divisor = span_len if loss_norm == "turn" else span_len ** 0.5
                    mi["loss_weights"] = [w / divisor for w in lw]

    for k, v in mi.items():
        if isinstance(v, torch.Tensor):
            mi[k] = v.tolist()

    return mi


def _batched(iterable, n: int):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_with_v1(
    model_name_or_path: str,
    data_dir: str,
    cutoff_len: int = 32768,
    output_dir: str = "./tokenized_dataset",
    train_mode: str = "last",
    num_workers: int = 1,
    think_mode: str = "last",
    loss_norm: str = "token",
) -> str:
    """Load → convert → tokenize → save.

    Args:
        train_mode: 'last' = 只学最后一个 assistant turn（默认）
                    'all'  = 每个 assistant turn 拆一条训练样本
        num_workers: 并行 token 化线程数
        think_mode: 'all'=保留所有reasoning, 'last'=只保留最后一轮 (默认), 'none'=全部删除
        loss_norm: 'token' = 默认 per-token 平均；'turn' = per-turn 平均（轮间等权）；
                   'sqrt' = 平方根长度归一化（每轮权重 ∝ sqrt(len)）
    """

    # ── 0. Load raw JSON ──
    raw_items = load_all_json_files(data_dir)

    # ── 1. V0: load tokenizer/processor ──
    from llamafactory.hparams import ModelArguments
    from llamafactory.model import load_tokenizer

    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    tokenizer_module = load_tokenizer(model_args)
    processor = tokenizer_module["processor"]

    # ── 2. Apply converter ──
    from v1_openai_converter import openai_agent_converter

    print(f"[Convert] Applying openai_agent converter to {len(raw_items)} items ...")
    converted = []
    for item in tqdm(raw_items, desc="Converting"):
        try:
            v1_sample = openai_agent_converter(item, think_mode=think_mode)
            converted.append(v1_sample)
        except Exception as e:
            print(f"\n[WARN] Converter error: {e}")

    for item in converted:
        if "tools" not in item:
            item["tools"] = json.dumps([])
    print(f"[Convert] {len(converted)} samples ready")

    # ── 3. Multi-turn split (optional) ──
    from llamafactory.v1.core.rendering import Renderer

    if train_mode == "all":
        expanded = _prefix_split_samples(converted)
        print(f"[Split] {len(converted)} conversations → {len(expanded)} training samples")
    else:
        expanded = converted
        print(f"[Split] Skipped (train_mode=last): {len(expanded)} samples, "
              f"each supervised on its last assistant turn only")

    # ── 4. Tokenize (parallel) ──
    print(f"[Tokenize] Rendering {len(expanded)} samples "
          f"(cutoff={cutoff_len}, workers={num_workers}, loss_norm={loss_norm}) ...")

    tokenized, dropped = [], 0
    # Use a list of dicts keyed by index so we can retain original order.
    indexed_results: dict[int, dict] = {}

    if num_workers <= 1:
        # ── Single-threaded ──
        renderer = Renderer(processor)
        for i, sample in enumerate(tqdm(expanded, desc="Tokenizing")):
            mi = _tokenize_one(sample, renderer, cutoff_len, loss_norm)
            if mi is not None:
                tokenized.append(mi)
            else:
                dropped += 1

    else:
        # ── Multi-threaded ──
        # HuggingFace tokenizer (Rust) is thread-safe and releases the GIL.
        import threading
        renderer = Renderer(processor)
        lock = threading.Lock()
        idx_iter = iter(range(len(expanded)))

        def _worker():
            while True:
                with lock:
                    try:
                        i = next(idx_iter)
                    except StopIteration:
                        return
                sample = expanded[i]
                mi = _tokenize_one(sample, renderer, cutoff_len, loss_norm)

                with lock:
                    if mi is not None:
                        indexed_results[i] = mi

        threads = [threading.Thread(target=_worker, daemon=True) for _ in range(num_workers)]
        for t in threads:
            t.start()

        with tqdm(total=len(expanded), desc="Tokenizing") as pbar:
            last_count = 0
            while any(t.is_alive() for t in threads):
                current = len(indexed_results)
                pbar.update(current - last_count)
                last_count = current
                import time
                time.sleep(0.1)
            pbar.update(len(indexed_results) - last_count)

        for t in threads:
            t.join(timeout=2)

        # Reconstruct in original order
        for i in sorted(indexed_results):
            tokenized.append(indexed_results[i])
        dropped = len(expanded) - len(tokenized)

    print(f"[Tokenize] {len(tokenized)} valid, {dropped} dropped")

    # ── 5. Save ──
    hf_dataset = HfDataset.from_list(tokenized)
    print(f"[Save] HfDataset: {hf_dataset}")
    os.makedirs(output_dir, exist_ok=True)
    hf_dataset.save_to_disk(output_dir)
    print(f"[Save] → {output_dir}")
    print(f"[Done] Use with V0:  tokenized_path: {output_dir}")
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V1 tokenize → V0 train")
    parser.add_argument("--model", required=True, help="HF model ID/path, e.g. Qwen/Qwen3.5-2B")
    parser.add_argument("--data_dir", required=True, help="Path to data_openai directory")
    parser.add_argument("--cutoff_len", type=int, default=32768)
    parser.add_argument("--output_dir", default="./tokenized_dataset")
    parser.add_argument("--train", dest="train_mode", default="last",
                        choices=["all", "last"],
                        help="all=每个assistant turn拆一条样本, last=只学最后一个assistant turn (默认)")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="并行 token 化线程数 (default: 32)")
    parser.add_argument("--think", dest="think_mode", default="last",
                        choices=["all", "last", "none"],
                        help="all=保留所有reasoning, last=只保留最后一轮 (默认), none=全部删除")
    parser.add_argument("--loss-norm", dest="loss_norm", default="token",
                        choices=["token", "turn", "sqrt"],
                        help="token=per-token平均 (默认), turn=per-turn平均 (轮间等权), "
                             "sqrt=平方根长度归一化 (每轮权重∝sqrt(len))")
    args = parser.parse_args()

    tokenize_with_v1(
        model_name_or_path=args.model,
        data_dir=args.data_dir,
        cutoff_len=args.cutoff_len,
        output_dir=args.output_dir,
        train_mode=args.train_mode,
        num_workers=args.num_workers,
        think_mode=args.think_mode,
        loss_norm=args.loss_norm,
    )
