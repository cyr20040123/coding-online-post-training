#!/usr/bin/env python3
"""
Step 1: 多轮对话清洗（去重 + 可选的前缀合并）

默认行为（不传 --merge）:
- 完全重复的对话去重（保留先出现的一条）
- 其余数据原样通过

开启 --merge 时额外:
- 前缀链检测: 若样本 A 的 messages 是样本 B 的 messages 的严格前缀，则 A 被 B 吸收
  （用于将 "messages[:cut]" 前缀拆分形式的数据合并回完整对话）
- 每组前缀链只保留最长样本 = 完整对话

开启 --relaxed 时（与 --merge 互斥）:
- 宽松合并: 先去掉所有 user: <system-reminder> 轮，得到"骨架"
- 骨架互为前缀的样本属于同一 lineage（即只允许由 <system-reminder> 轮造成的分叉被合并）
- 每个 lineage 以骨架最长的样本为基底，把全部历史 <system-reminder> 轮按原位置插回
- 非 <system-reminder> 造成的分叉（骨架在某处内容不同）独立保留为多个 item

--keep-longest（可与默认/--merge/--relaxed 任意模式组合）:
- 在所选模式处理完成后，每个来源文件仅保留消息数最长的一条对话
- 用于"每实例一条完整轨迹"的最终产物形态

输入: data_openai 目录（每个 JSON 文件是一个 item 列表，OpenAI 格式）
输出: 一行一条对话的 JSONL（含 conversation_id，可检视）

用法:
    # 仅去重（默认）
    python merge_split_to_multiturn.py \
        --input /path/to/data_openai \
        --output ./data/multiturn_merged.jsonl

    # 去重 + 前缀合并
    python merge_split_to_multiturn.py \
        --input /path/to/data_openai \
        --output ./data/multiturn_merged.jsonl \
        --merge

    # 宽松合并（sysrem 分叉合并、其余分叉独立保留）
    python merge_split_to_multiturn.py \
        --input /path/to/data_openai \
        --output ./data/multiturn_merged.jsonl \
        --relaxed
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


SYSREM_MARKER = "<system-reminder>"


def _is_sysrem_turn(msg: dict) -> bool:
    """是否为注入的 <system-reminder> user 轮。"""
    return msg.get("role") == "user" and SYSREM_MARKER in (msg.get("content", "") or "")


def _split_skeleton(messages: list[dict]) -> tuple[list[dict], list[tuple[int, dict]]]:
    """把消息列表拆成 (骨架, [(骨架偏移, sysrem消息), ...])。

    骨架 = 去掉全部 <system-reminder> user 轮后的消息序列；
    骨架偏移 = 该 sysrem 轮之前骨架消息的条数（用于插回原位）。
    """
    skeleton: list[dict] = []
    sysrem: list[tuple[int, dict]] = []
    for m in messages:
        if _is_sysrem_turn(m):
            sysrem.append((len(skeleton), m))
        else:
            skeleton.append(m)
    return skeleton, sysrem


def _rebuild_messages(skeleton: list[dict],
                      sysrem_entries: list[tuple[int, dict]]) -> list[dict]:
    """按骨架偏移把 sysrem 轮插回骨架，重建完整消息列表。

    同偏移且同签名的轮只保留一份；不同偏移（即使内容相同）全部保留。
    """
    pending: dict[int, list[dict]] = defaultdict(list)
    seen_keys: set[tuple[int, str]] = set()
    for offset, msg in sysrem_entries:
        key = (offset, _msg_signature(msg))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        pending[offset].append(msg)
    out: list[dict] = []
    for k, m in enumerate(skeleton):
        out.extend(pending.get(k, []))
        out.append(m)
    for k in sorted(pending):
        if k >= len(skeleton):
            out.extend(pending[k])
    return out


def load_items(input_path: str) -> list[tuple[str, dict]]:
    """加载 data_openai 所有 JSON 文件，返回 [(来源文件, item), ...]。

    来源文件标识：目录输入时为相对输入目录的路径（不同子目录的同名文件
    不会冲突），单文件输入时为文件名（与旧行为一致）。
    """
    items = []
    if os.path.isdir(input_path):
        json_files = sorted(glob.glob(os.path.join(input_path, "**/*.json"), recursive=True))
        base_dir = input_path
    elif os.path.isfile(input_path):
        json_files = [input_path]
        base_dir = os.path.dirname(input_path)
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"[Load] Found {len(json_files)} JSON files")
    for fp in json_files:
        src = os.path.relpath(fp, base_dir)
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                items.append((src, item))
        elif isinstance(data, dict):
            items.append((src, data))
    print(f"[Load] Total: {len(items)} items")
    return items


# ──────────────────────────────────────────────────────────────────────────────
# relaxed merge
# ──────────────────────────────────────────────────────────────────────────────

def relaxed_merge(items: list[tuple[str, dict]]) -> tuple[list[tuple[str, dict, int]], dict]:
    """宽松合并: 只允许由 <system-reminder> 轮造成的分叉合并，其余分叉独立保留。

    判据: 去掉 sysrem 轮后的"骨架"互为严格前缀（或完全相同）的样本属于同一
    lineage；骨架在某处内容不同即为真分叉，各自独立保留。

    每个 lineage 以骨架最长的样本为基底，把全部成员的历史 sysrem 轮按原位置
    插回，得到一条完整对话。

    返回 ([(src_file, item, base_idx), ...], stats)。
    """
    skels: dict[int, list[dict]] = {}
    sysrems: dict[int, list[tuple[int, dict]]] = {}
    orphan: list[int] = []   # 空 messages 或全是 sysrem 轮的 item，原样保留
    for idx, (_, item) in enumerate(items):
        msgs = item.get("messages")
        if not isinstance(msgs, list) or not msgs:
            orphan.append(idx)
            continue
        skels[idx], sysrems[idx] = _split_skeleton(msgs)
        if not skels[idx]:
            orphan.append(idx)

    # 按骨架首条消息签名分组；组内按骨架长度降序做 lineage 聚类
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in skels:
        groups[_msg_signature(skels[idx][0])].append(idx)

    # 预计算每条骨架的签名序列（每条消息只哈希一次），
    # 聚类时的前缀/相等判断退化为 tuple 比较，避免 O(n^2) 重复哈希
    skel_sigs: dict[int, tuple[str, ...]] = {
        idx: tuple(_msg_signature(m) for m in skels[idx]) for idx in skels
    }

    lineage_of: dict[int, int] = {}
    lineages: dict[int, list[int]] = defaultdict(list)
    n_multi_compat = 0
    for members in groups.values():
        members_desc = sorted(members, key=lambda i: (-len(skels[i]), i))
        for pos_i, i in enumerate(members_desc):
            a = skel_sigs[i]
            parents = []
            for j_idx in range(pos_i):
                j = members_desc[j_idx]
                b = skel_sigs[j]
                if len(b) > len(a):
                    if all(x == y for x, y in zip(a, b)):
                        parents.append(j)
                elif a == b:
                    parents.append(j)
            if parents:
                if len(parents) > 1:
                    n_multi_compat += 1
                root = lineage_of[parents[0]]   # 归属到最长的兼容者
            else:
                root = i
            lineage_of[i] = root
            lineages[root].append(i)

    # 每个 lineage: 最长骨架为基底，插回全部成员的历史 sysrem 轮
    out_items: list[tuple[str, dict, int]] = []
    n_absorbed = 0
    n_sysrem_inserted = 0
    for root in sorted(lineages, key=lambda r: min(lineages[r])):
        member_idxs = sorted(lineages[root])
        base = max(member_idxs, key=lambda i: (len(skels[i]), -i))
        all_sysrem = [entry for m in member_idxs for entry in sysrems[m]]
        merged_msgs = _rebuild_messages(skels[base], all_sysrem)
        out_item = dict(items[base][1])
        out_item["messages"] = merged_msgs
        out_item["merge_info"] = {
            "mode": "relaxed",
            "absorbed_indices": [m for m in member_idxs if m != base],
            "n_sysrem_inserted": len(merged_msgs) - len(skels[base]),
        }
        out_items.append((items[base][0], out_item, base))
        n_absorbed += len(member_idxs) - 1
        n_sysrem_inserted += len(merged_msgs) - len(skels[base])

    # 无骨架的 item 原样输出
    for idx in orphan:
        src_file, item = items[idx]
        out_item = dict(item)
        out_item["merge_info"] = {"mode": "relaxed", "absorbed_indices": [],
                                  "n_sysrem_inserted": 0}
        out_items.append((src_file, out_item, idx))

    n_fork_groups = sum(
        1 for members in groups.values()
        if len({lineage_of[i] for i in members}) > 1
    )
    stats = {
        "n_input": len(items),
        "n_output": len(out_items),
        "n_absorbed": n_absorbed,
        "n_sysrem_inserted": n_sysrem_inserted,
        "n_fork_groups": n_fork_groups,
        "n_multi_compat": n_multi_compat,
    }
    return out_items, stats


def keep_longest_per_file(out_items: list[tuple[str, dict, int]]) -> list[tuple[str, dict, int]]:
    """每个来源文件仅保留消息数最长的一条对话（并列时保留先出现的）。

    对任意模式产出的 (src_file, item, id_idx) 列表做按文件归约，
    输出顺序与首次出现顺序一致。
    """
    best: dict[str, tuple[str, dict, int]] = {}
    for entry in out_items:
        src_file, item, _ = entry
        cur_len = len(item.get("messages") or [])
        if src_file not in best or cur_len > len(best[src_file][1].get("messages") or []):
            best[src_file] = entry
    return list(best.values())


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Merge prefix-split multi-turn conversations")
    parser.add_argument("--input", "-i", required=True, help="data_openai 目录或单个 JSON 文件")
    parser.add_argument("--output", "-o", required=True, help="输出 JSONL 路径")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--merge", action="store_true", default=False,
                      help="开启前缀匹配合并（默认关闭，仅做完全重复去重）")
    mode.add_argument("--relaxed", action="store_true", default=False,
                      help="宽松合并: 只允许 <system-reminder> 轮造成的分叉合并，其余分叉独立保留")
    parser.add_argument("--keep-longest", action="store_true", default=False,
                        help="每个来源文件仅保留最长的一条对话（三种模式均可启用）")
    args = parser.parse_args()

    items = load_items(args.input)

    if args.relaxed:
        out_items, stats = relaxed_merge(items)
    else:
        # 按首条消息签名分组，减少前缀比较的候选集
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, (_, item) in enumerate(items):
            msgs = item.get("messages")
            if not isinstance(msgs, list) or not msgs:
                continue
            groups[_msg_signature(msgs[0])].append(idx)

        absorbed: set[int] = set()   # 被吸收的前缀样本索引
        duplicate_of: dict[int, int] = {}
        n_duplicates = 0
        n_prefix_merged = 0

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
                            n_duplicates += 1
                    elif args.merge and _is_prefix(msgs_i, msgs_j):
                        absorbed.add(i)
                        n_prefix_merged += 1
                        break  # i 已确认被吸收，无需继续

        # 未被吸收的（最长）样本
        out_items = [(src_file, item, idx)
                     for idx, (src_file, item) in enumerate(items)
                     if idx not in absorbed]

    keep_longest_stats: tuple[int, int] | None = None
    if args.keep_longest:
        n_before = len(out_items)
        out_items = keep_longest_per_file(out_items)
        keep_longest_stats = (n_before, len(out_items))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for src_file, item, id_idx in out_items:
            out = dict(item)
            out["conversation_id"] = f"conv-{id_idx:05d}"
            out["source_file"] = src_file
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    if args.relaxed:
        print(f"[Relaxed] {stats['n_input']} items → {stats['n_output']} conversations")
        print(f"          absorbed: {stats['n_absorbed']}, sysrem turns inserted: {stats['n_sysrem_inserted']}")
        print(f"          fork groups (non-sysrem divergence): {stats['n_fork_groups']}")
        print(f"          items compatible with multiple lineages: {stats['n_multi_compat']}")
    else:
        print(f"[Merge] {len(items)} items → {written} complete conversations")
        print(f"        duplicates removed: {n_duplicates}")
        print(f"        prefix-merged:      {n_prefix_merged} (merge={'on' if args.merge else 'off'})")
    if keep_longest_stats is not None:
        print(f"[KeepLongest] {keep_longest_stats[0]} → {keep_longest_stats[1]} conversations (longest per source file)")
    print(f"[Save] → {args.output}")


if __name__ == "__main__":
    main()
