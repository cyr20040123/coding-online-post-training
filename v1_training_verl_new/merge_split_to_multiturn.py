#!/usr/bin/env python3
"""
Step 1: 拆分形式的多轮对话 → 合并为完整多轮对话

将前缀拆分形式的数据（同源对话被切成 "messages[:cut]" 前缀链）合并回完整对话。
- 前缀链检测: 若样本 A 的 messages 是样本 B 的 messages 的严格前缀，则 A 被 B 吸收
- 合并规则: 每组前缀链只保留最长样本 = 完整对话
- 幂等: 无前缀关系的数据（如 data_openai 已是完整对话）原样通过

输入: data_openai 目录（每个 JSON 文件是一个 item 列表，OpenAI 格式）
输出: 一行一条完整多轮对话的 JSONL（含 conversation_id，可检视）

用法:
    python merge_split_to_multiturn.py \
        --input /path/to/data_openai \
        --output ./data/multiturn_merged.jsonl
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _msg_signature(msg: dict) -> str:
    """消息的规范化签名，用于前缀匹配（只比较角色+内容，忽略无关字段）。"""
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        tool_calls = [
            {"name": tc.get("function", tc).get("name", ""),
             "arguments": tc.get("function", tc).get("arguments", "")}
            for tc in tool_calls
        ]
    norm = {
        "role": msg.get("role", ""),
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
    }
    raw = json.dumps(norm, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _is_prefix(short: list[dict], long: list[dict]) -> bool:
    """判断 short 的 messages 是否是 long 的 messages 的严格前缀。"""
    if len(short) >= len(long):
        return False
    for m_s, m_l in zip(short, long):
        if _msg_signature(m_s) != _msg_signature(m_l):
            return False
    return True


def load_items(input_path: str) -> list[tuple[str, dict]]:
    """加载 data_openai 所有 JSON 文件，返回 [(来源文件, item), ...]。"""
    items = []
    if os.path.isdir(input_path):
        json_files = sorted(glob.glob(os.path.join(input_path, "**/*.json"), recursive=True))
    elif os.path.isfile(input_path):
        json_files = [input_path]
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"[Load] Found {len(json_files)} JSON files")
    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                items.append((os.path.basename(fp), item))
        elif isinstance(data, dict):
            items.append((os.path.basename(fp), data))
    print(f"[Load] Total: {len(items)} items")
    return items


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Merge prefix-split multi-turn conversations")
    parser.add_argument("--input", "-i", required=True, help="data_openai 目录或单个 JSON 文件")
    parser.add_argument("--output", "-o", required=True, help="输出 JSONL 路径")
    args = parser.parse_args()

    items = load_items(args.input)

    # 按首条消息签名分组，减少前缀比较的候选集
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (_, item) in enumerate(items):
        msgs = item.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue
        groups[_msg_signature(msgs[0])].append(idx)

    absorbed: set[int] = set()   # 被吸收的前缀样本索引
    duplicate_of: dict[int, int] = {}

    for sig, members in groups.items():
        members_sorted = sorted(members, key=lambda i: len(items[i][1]["messages"]))
        # 组内两两检查: 短样本是否是长样本的前缀
        for i_idx in range(len(members_sorted)):
            i = members_sorted[i_idx]
            if i in absorbed:
                continue
            for j_idx in range(i_idx + 1, len(members_sorted)):
                j = members_sorted[j_idx]
                msgs_i = items[i][1]["messages"]
                msgs_j = items[j][1]["messages"]
                if len(msgs_i) == len(msgs_j):
                    if all(_msg_signature(a) == _msg_signature(b) for a, b in zip(msgs_i, msgs_j)):
                        # 完全重复的对话，保留先出现的
                        duplicate_of[j] = i
                        absorbed.add(j)
                elif _is_prefix(msgs_i, msgs_j):
                    absorbed.add(i)
                    break  # i 已确认被吸收，无需继续

    # 输出未被吸收的（最长）样本
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for idx, (src_file, item) in enumerate(items):
            if idx in absorbed:
                continue
            out = dict(item)
            out["conversation_id"] = f"conv-{idx:05d}"
            out["source_file"] = src_file
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"[Merge] {len(items)} items → {written} complete conversations")
    print(f"        absorbed (prefix/duplicate): {len(absorbed)}")
    print(f"[Save] → {args.output}")


if __name__ == "__main__":
    main()
