#!/usr/bin/env python3
"""
Merge FSDP sharded verl checkpoint to HuggingFace LoRA adapter (safetensors).

Usage:
  python merge_checkpoint_to_hf.py \
    --input checkpoints/jiuwen-sft-verl/qwen3.5-2b-lora64-sp4/global_step_180 \
    --output /tmp/lora-adapter
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

import torch
from safetensors.torch import save_file


def merge_sharded_checkpoint(input_dir: str, output_dir: str):
    """Load all FSDP shards, merge DTensor shards, and save as HF format."""

    model_files = sorted(
        f for f in os.listdir(input_dir)
        if re.match(r"model_world_size_\d+_rank_\d+\.pt$", f)
    )
    if not model_files:
        print(f"ERROR: No model shards found in {input_dir}")
        sys.exit(1)

    num_ranks = len(model_files)
    print(f"Found {num_ranks} shards, loading...")

    shards = {}
    for i, fname in enumerate(model_files):
        path = os.path.join(input_dir, fname)
        print(f"  [{i+1}/{num_ranks}] {fname}")
        shards[i] = torch.load(path, map_location="cpu", weights_only=False)

    # Collect all unique keys
    all_keys = sorted(shards[0].keys())
    print(f"Total keys per shard: {len(all_keys)}")

    # Extract LoRA keys only (lora_A, lora_B)
    lora_keys = [k for k in all_keys if 'lora_A' in k or 'lora_B' in k]
    other_keys = [k for k in all_keys if k not in lora_keys]
    print(f"LoRA adapter keys: {len(lora_keys)}")
    print(f"Non-LoRA (base model) keys: {len(other_keys)}")

    # 剔除视觉塔的 LoRA keys（纯文本 SFT 训练时 vision 被冻结，这些是未训练的初始权重；
    # vLLM 对 vision LoRA 支持有限，剔除后与 llama 的 adapter 模块集一致）
    visual_keys = [k for k in lora_keys if '.visual.' in k]
    lora_keys = [k for k in lora_keys if '.visual.' not in k]
    if visual_keys:
        print(f"Skipped vision-tower keys (frozen, untrained): {len(visual_keys)}")

    # 从实际 keys 动态提取模块名（保证 adapter_config.json 与 safetensors 一致）
    # key 形如: base_model...layers.N.linear_attn.in_proj_a.lora_A[.default].weight
    # 模块名 = lora_A 之前紧邻的一段（如 q_proj / in_proj_a）
    target_modules = []
    for k in lora_keys:
        if 'lora_A' not in k:
            continue
        m = re.search(r'\.([A-Za-z0-9_]+)\.lora_A', k)
        if m and m.group(1) not in target_modules:
            target_modules.append(m.group(1))
    print(f"Detected target_modules ({len(target_modules)}): {target_modules}")

    # Merge LoRA keys from all shards
    merged = OrderedDict()
    for key in lora_keys:
        # Each rank holds a DTensor with .to_local() giving local shard
        local_tensors = []
        for rank in range(num_ranks):
            dt = shards[rank][key]
            local_tensors.append(dt.to_local())

        # Stack along dim 0 (FSDP default sharding for LoRA)
        merged[key] = torch.cat(local_tensors, dim=0)

    print(f"Merged {len(merged)} LoRA param tensors")

    # 剥掉 PEFT adapter 名段 ".default"：
    #   verl 保存: ...linear_attn.in_proj_a.lora_A.default.weight
    #   vLLM 要求: ...linear_attn.in_proj_a.lora_A.weight
    #   （vllm/lora/utils.py parse_fine_tuned_lora_name 只接受 lora_A/B 紧邻 weight 的 key）
    renamed = OrderedDict()
    for key, tensor in merged.items():
        clean = key.replace(".default.weight", ".weight")
        if clean != key:
            renamed[clean] = tensor
        else:
            renamed[key] = tensor
    merged = renamed
    print(f"Renamed keys (stripped '.default'): {sum(1 for k in merged if '.default.' not in k)}")

    # Save as safetensors
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(merged, save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Saved: {save_path} ({file_size_mb:.1f} MB)")

    # Copy config files from huggingface subdirectory
    hf_dir = os.path.join(input_dir, "huggingface")
    if os.path.isdir(hf_dir):
        import shutil
        for fname in os.listdir(hf_dir):
            src = os.path.join(hf_dir, fname)
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
        print(f"Copied config files from {hf_dir}")

    # Write adapter_config.json
    # Use the lora_train_meta.json for rank/alpha info
    meta_path = os.path.join(input_dir, "lora_train_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        r = meta.get("r", 64)
        lora_alpha = meta.get("lora_alpha", 64)
        task_type = meta.get("task_type", "CAUSAL_LM")
    else:
        r, lora_alpha, task_type = 64, 64, "CAUSAL_LM"

    adapter_config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": "Qwen/Qwen3.5-2B",
        "bias": "none",
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "middle_layers": None,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": r,
        "rank_pattern": {},
        "target_modules": target_modules,
        "task_type": task_type,
        "use_dora": False,
        "use_rslora": False,
    }
    config_path = os.path.join(output_dir, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
    print(f"Saved: {config_path}")

    print("\nDone! Output directory:", output_dir)
    print("Contents:", sorted(os.listdir(output_dir)))


def main():
    parser = argparse.ArgumentParser(
        description="Merge verl FSDP sharded checkpoint to HF LoRA adapter"
    )
    parser.add_argument("--input", "-i", required=True, help="Checkpoint directory path")
    parser.add_argument("--output", "-o", required=True, help="Output directory for HF adapter")
    args = parser.parse_args()
    merge_sharded_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
