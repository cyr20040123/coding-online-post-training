#!/usr/bin/env python3
"""
Step 2: 完整多轮对话 → loss-norm 处理后的 VeRL Parquet

输入: Step 1 产出的合并多轮对话 JSONL（OpenAI 格式）
处理:
  1. OpenAI → Qwen 文本格式（tool_calls 内联 XML、tool 响应 → <tool_response> user 消息）
  2. 逐消息独立 tokenize（与 qwen_multiturn_sft_dataset.py 同一套逻辑）
  3. 按每条 assistant 消息的响应长度计算连续 loss_mask（三种归一化模式）
输出 Parquet 列:
  messages      — 转换后的多轮对话（检视用）
  input_ids / attention_mask / position_ids — 训练用
  loss_mask     — 连续值 mask（已含归一化，检视用）
  turn_lengths  — 每轮 assistant 响应 token 长度（检视归一化依据）

loss-norm 模式:
  token — 每 token 权重 1                （每轮总权重 = len）
  sqrt  — 每 token 权重 1/sqrt(len)      （每轮总权重 = sqrt(len)）
  turn  — 每 token 权重 1/len            （每轮总权重 = 1）

用法:
    python multiturn_to_verl_norm.py \
        --input ./data/multiturn_merged.jsonl \
        --output ./data/sft_train_norm.parquet \
        --model Qwen/Qwen3.5-2B \
        --loss-norm turn
"""

import argparse
import json
import logging
import os
import re
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. OpenAI → Qwen 文本格式转换
# ──────────────────────────────────────────────────────────────────────────────

def format_tool_call_qwen_xml(name: str, arguments: dict) -> str:
    """单个工具调用 → Qwen XML 格式（与 convert_data_to_verl.py 一致）。"""
    lines = ["<tool_call>", f"<function={name}>"]
    for param_name, param_value in arguments.items():
        if isinstance(param_value, (dict, list)):
            value_str = json.dumps(param_value, ensure_ascii=False, indent=2)
        else:
            value_str = str(param_value)
        lines.append(f"<parameter={param_name}>")
        lines.append(value_str)
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def _parse_tool_call_args(args):
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


_THINK_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*", re.DOTALL)


def extract_think_from_content(content: str) -> tuple[str, str]:
    """从 content 开头的 <think> 块提取思考过程，返回 (reasoning, cleaned_content)。"""
    match = _THINK_RE.match(content)
    if not match:
        return "", content
    return match.group(1).strip(), content[match.end():]


def convert_message_openai(msg: dict) -> dict:
    """OpenAI 格式消息 → Qwen 文本格式消息。

    返回 {"role", "content", ["reasoning_content"]}。
    - assistant: reasoning_content 字段或 content 内嵌 <think> → reasoning_content；
      tool_calls 数组 → XML 内联到 content
    - tool: role → user，内容包 <tool_response>
    - system/user: 原样
    """
    role = msg.get("role", "")
    content = msg.get("content", "") or ""
    if not isinstance(content, str):
        # 多模态 list content：只取 text 部分
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        content = "".join(parts)

    if role == "assistant":
        reasoning = (msg.get("reasoning_content") or "").strip()
        text = content
        if not reasoning and "<think>" in text:
            reasoning, text = extract_think_from_content(text)
        text = text.strip()

        # 工具调用 XML 内联
        tool_calls = msg.get("tool_calls") or []
        xml_parts = []
        for tc in tool_calls:
            fn = tc.get("function", tc)
            fn_name = fn.get("name", "")
            fn_args = _parse_tool_call_args(fn.get("arguments", {}))
            xml_parts.append(format_tool_call_qwen_xml(fn_name, fn_args))

        if xml_parts:
            text = "\n".join([t for t in [text, *xml_parts] if t])

        out = {"role": "assistant", "content": text}
        if reasoning:
            out["reasoning_content"] = reasoning
        return out

    if role == "tool":
        return {"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"}

    return {"role": role, "content": content}


# ──────────────────────────────────────────────────────────────────────────────
# 2. 逐消息 tokenize（与 qwen_multiturn_sft_dataset.py 同一套逻辑）
# ──────────────────────────────────────────────────────────────────────────────

def normalize_token_ids(tokenized_output) -> list[int]:
    """Normalize tokenizer outputs into a flat ``list[int]``.

    Handles Transformers 4/5 differences: ``apply_chat_template(tokenize=True)``
    may return ``list[int]`` or a ``BatchEncoding``/mapping with ``input_ids``.
    （与 verl.utils.tokenizer.normalize_token_ids 逻辑一致）
    """
    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], (list, tuple)):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like, got {type(token_ids).__name__}")
    return [int(t) for t in token_ids]


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", item.get("value", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def tokenize_single_message(tokenizer, msg: dict) -> list[int]:
    """单条消息 tokenize，处理 Qwen 模板约束（至少一条 user、system 必须开头）。"""
    try:
        return normalize_token_ids(
            tokenizer.apply_chat_template([msg], add_generation_prompt=False, tokenize=True)
        )
    except Exception:
        pass

    if msg["role"] == "system":
        content = _flatten_content(msg.get("content", ""))
        text = f"<|im_start|>system\n{content}<|im_end|>\n"
        return normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))

    dummy = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    try:
        dummy_prefix = normalize_token_ids(
            tokenizer.apply_chat_template(dummy, add_generation_prompt=False, tokenize=True)
        )
        combined = normalize_token_ids(
            tokenizer.apply_chat_template(dummy + [msg], add_generation_prompt=False, tokenize=True)
        )
        return combined[len(dummy_prefix):]
    except Exception:
        content = _flatten_content(msg.get("content", ""))
        text = f"<|im_start|>{msg['role']}\n{content}<|im_end|>\n"
        return normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))


# ──────────────────────────────────────────────────────────────────────────────
# 3. loss_mask 归一化
# ──────────────────────────────────────────────────────────────────────────────

def _mask_for_mode(turn_len: int, mode: str) -> float:
    """每 token 权重：token → 1；sqrt → 1/√len；turn → 1/len。"""
    if turn_len <= 0:
        return 0.0
    if mode == "turn":
        return 1.0 / turn_len
    if mode == "sqrt":
        return 1.0 / (turn_len ** 0.5)
    return 1.0


def _turn_segments(messages: list[dict]) -> list[list[dict]]:
    """按轮切分：每轮 = 上一个 assistant 之后到下一个 assistant 为止的消息段。"""
    segments: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages:
        current.append(msg)
        if msg.get("role") == "assistant":
            segments.append(current)
            current = []
    return segments


def build_sample(tokenizer, messages: list[dict], loss_norm: str, supervise: str = "all") -> dict:
    """一条完整多轮对话 → {input_ids, attention_mask, position_ids, loss_mask,
    turn_lengths, turn_offsets}。

    不截断：存完整对话。超长处理交给训练时 dataset 类的滑动窗口
    （truncation 策略 = 按轮边界拆窗，保留较新轮次，头轮不丢失）。
    turn_offsets: 每轮 segment 在 token 序列中的 [start, end) 偏移，供滑动窗口使用。

    supervise: 'all'  = 所有 assistant turn 都监督（默认，verl 语义）
               'last' = 只监督最后一个 assistant turn（对齐 llama --train last 语义；
                        配合关闭滑动窗口使用）
    """
    segments = _turn_segments(messages)
    last_seg_idx = len(segments) - 1

    all_ids: list[int] = []
    all_mask: list[float] = []
    turn_lengths: list[int] = []
    turn_offsets: list[list[int]] = []

    for seg_idx, seg in enumerate(segments):
        seg_start = len(all_ids)
        for msg in seg:
            msg_tokens = tokenize_single_message(tokenizer, msg)
            all_ids.extend(msg_tokens)
            if msg.get("role") == "assistant":
                empty_tokens = tokenize_single_message(tokenizer, {"role": "assistant", "content": ""})
                gen_len = len(empty_tokens)
                turn_len = max(0, len(msg_tokens) - gen_len)
                turn_lengths.append(turn_len)
                if supervise == "last" and seg_idx != last_seg_idx:
                    # 非最后轮：全部 mask（不监督）
                    all_mask.extend([0.0] * len(msg_tokens))
                else:
                    w = _mask_for_mode(turn_len, loss_norm)
                    all_mask.extend([0.0] * gen_len + [w] * (len(msg_tokens) - gen_len))
            else:
                all_mask.extend([0.0] * len(msg_tokens))
        turn_offsets.append([seg_start, len(all_ids)])

    n = len(all_ids)
    return {
        "input_ids": all_ids,
        "attention_mask": [1] * n,
        "position_ids": list(range(n)),
        "loss_mask": all_mask,
        "turn_lengths": turn_lengths,
        "turn_offsets": turn_offsets,
    }


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def _is_multimodal(raw_messages: list) -> bool:
    """检测多模态数据：任何消息的 content 是 list（OpenAI image_url 块格式）。"""
    for m in raw_messages:
        content = m.get("content")
        if isinstance(content, list):
            return True
    return False


def _ends_with_assistant(raw_messages: list) -> bool:
    """检测对话是否以 assistant 结尾（才有可监督的响应）。"""
    return bool(raw_messages) and raw_messages[-1].get("role") == "assistant"


def main():
    parser = argparse.ArgumentParser(description="Multi-turn conversations → loss-norm VeRL Parquet")
    parser.add_argument("--input", "-i", required=True, help="Step 1 产出的合并 JSONL")
    parser.add_argument("--output", "-o", required=True, help="输出 Parquet 路径")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B", help="tokenizer 模型")
    parser.add_argument("--loss-norm", dest="loss_norm", default="token",
                        choices=["token", "turn", "sqrt"],
                        help="token=per-token (默认), turn=per-turn 等权, sqrt=平方根归一化")
    parser.add_argument("--max-samples", type=int, default=-1, help="限制样本数 (-1 = 全部)")
    parser.add_argument("--no-filter", dest="no_filter", action="store_true", default=False,
                        help="不过滤（保持原行为）；默认过滤多模态及无 assistant 结尾的数据")
    parser.add_argument("--supervise", dest="supervise", default="all",
                        choices=["all", "last"],
                        help="all=所有assistant turn都监督 (默认), last=只监督最后一个assistant turn")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error("Input not found: %s", args.input)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 读取合并 JSONL
    conversations = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skip invalid line %d", line_no)

    if args.max_samples > 0:
        conversations = conversations[: args.max_samples]
    logger.info("Loaded %d conversations (filter=%s)", len(conversations),
                "off" if args.no_filter else "on")

    rows = []
    skipped = 0
    filtered_multimodal = 0
    filtered_no_assistant = 0
    total_tokens = 0
    for conv in conversations:
        raw_messages = conv.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            skipped += 1
            continue

        # ── 过滤（默认开启，--no-filter 关闭）──
        if not args.no_filter:
            if _is_multimodal(raw_messages):
                filtered_multimodal += 1
                continue
            if not _ends_with_assistant(raw_messages):
                filtered_no_assistant += 1
                continue

        # OpenAI → Qwen 文本格式（system 保留，agent 数据依赖 system prompt）
        converted = [convert_message_openai(m) for m in raw_messages]

        sample = build_sample(tokenizer, converted, args.loss_norm, args.supervise)
        rows.append({
            "messages": converted,
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "position_ids": sample["position_ids"],
            "loss_mask": sample["loss_mask"],
            "turn_lengths": sample["turn_lengths"],
            "turn_offsets": sample["turn_offsets"],
        })
        total_tokens += len(sample["input_ids"])

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False, engine="pyarrow")

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    logger.info("Saved Parquet: %s (%.1f MB, %d rows)", args.output, size_mb, len(df))
    logger.info("Loss norm mode: %s, supervise: %s", args.loss_norm, args.supervise)
    logger.info("Total tokens: %d", total_tokens)
    logger.info("Filtered: multimodal=%d, no_assistant_end=%d, empty/invalid=%d",
                filtered_multimodal, filtered_no_assistant, skipped)

    # 检视: 第一条样本的每轮长度与总权重
    if rows:
        tl = rows[0]["turn_lengths"]
        weights = [round(_mask_for_mode(l, args.loss_norm) * l, 4) for l in tl]
        logger.info("--- Sample 0 inspection ---")
        logger.info("  turn_lengths: %s", tl)
        logger.info("  per-turn total weights (len × w): %s", weights)
        for i, m in enumerate(rows[0]["messages"][:6]):
            c = str(m.get("content", ""))[:100].replace("\n", "\\n")
            logger.info("  [%d] role=%s content=%s...", i, m["role"], c)


if __name__ == "__main__":
    main()
