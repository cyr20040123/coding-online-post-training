#!/usr/bin/env python3
"""
OpenAI 原生格式 → LLaMAFactory V1 内部消息格式 转换器。

直接从 data_openai 的 OpenAI 格式（tool_calls 数组、reasoning_content 字段）
转换为 V1 Message 格式。无需经过 data_convert_openai2swift.py 中间步骤。

V1 数据流:
    OpenAI 原始 → 本转换器 → V1 Message
                          → V1 Renderer._to_hf_messages() → OpenAI format
                          → tokenizer.apply_chat_template() → token IDs

在 dataset_info.yaml 中通过 ``converter: openai_agent`` 引用。
"""

import json
from typing import Any

from llamafactory.v1.plugins.data_plugins.converter import DataConverterPlugin
from llamafactory.v1.utils.types import SFTSample


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _is_system_reminder(content: str) -> bool:
    """Detect harness-injected <system-reminder> noise."""
    return bool(content and "<system-reminder>" in content)


def _parse_tool_call_args(args) -> dict:
    """Parse tool_call arguments to a dict."""
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


# ---------------------------------------------------------------------------
# helper: extract <think>...</think> from content
# ---------------------------------------------------------------------------
import re
import warnings

_THINK_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*", re.DOTALL)


def _extract_think_from_content(content: str) -> tuple[str, str]:
    """Extract leading ``<think>...</think>`` block from content string.

    Returns ``(reasoning_text, cleaned_content)``.
    """
    match = _THINK_RE.match(content)
    if not match:
        return "", content
    reasoning = match.group(1).strip()
    cleaned = content[match.end() :]
    return reasoning, cleaned


# ---------------------------------------------------------------------------
# converter
# ---------------------------------------------------------------------------

@DataConverterPlugin("openai_agent").register()
def openai_agent_converter(raw_sample: dict[str, Any], think_mode: str = "last") -> SFTSample:
    """Convert OpenAI native agent format → V1 SFT sample.

    Args:
        raw_sample: OpenAI-format sample with messages + optional tools.
        think_mode:
            'all'  — 保留所有 assistant 的 reasoning_content（原始行为）
            'last' — 只保留最后一轮 assistant 的 reasoning_content（默认）
            'none' — 删除所有 reasoning_content

    Input (data_openai format):
        {
            "messages": [
                {"role": "system", "content": "You are..."},
                {"role": "user", "content": "task..."},
                {"role": "user", "content": "<system-reminder>..."},     ← discarded
                {"role": "assistant", "reasoning_content": "Need to search",
                 "content": "Let me look.", "tool_calls": [
                    {"id": "...", "function": {"name": "grep", "arguments": "{...}"}}
                 ]},
                {"role": "tool", "tool_call_id": "...", "content": "result..."},
                {"role": "assistant", "reasoning_content": "...",
                 "content": "final answer"}
            ]
        }

    Output (V1 internal Message format):
        {
            "messages": [
                {"role": "system", "content": [{"type":"text","value":"..."}], "loss_weight":0.0},
                {"role": "user", "content": [{"type":"text","value":"..."}], "loss_weight":0.0},
                {"role": "assistant", "content": [
                    {"type":"reasoning","value":"Need to search"},
                    {"type":"text","value":"Let me look."},
                    {"type":"tool_call","value":"{\\"name\\":\\"grep\\",...}"}
                ], "loss_weight":1.0},
                {"role": "tool", "content": [{"type":"text","value":"result..."}], "loss_weight":0.0},
                {"role": "assistant", "content": [
                    {"type":"reasoning","value":"..."},
                    {"type":"text","value":"final answer"}
                ], "loss_weight":1.0}
            ]
        }
    """
    messages = []
    raw_messages = raw_sample.get("messages", [])

    # 确定最后一轮 assistant 的位置（用于 --think last）
    last_assistant_idx = -1
    if think_mode == "last":
        for i in range(len(raw_messages) - 1, -1, -1):
            if raw_messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

    for idx, msg in enumerate(raw_messages):
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        # ── system ──
        if role == "system":
            messages.append({
                "role": "system",
                "content": [{"type": "text", "value": content}],
                "loss_weight": 0.0,
            })
            continue

        # ── user ──
        if role == "user":
            # TODO: 暂时保留 <system-reminder>，观察训练效果
            # if _is_system_reminder(content):
            #     continue
            messages.append({
                "role": "user",
                "content": [{"type": "text", "value": content}],
                "loss_weight": 0.0,
            })
            continue

        # ── assistant ──
        if role == "assistant":
            # 0. 检测 content 中不规范的内嵌 <think>，提取到 reasoning_content
            if content and "<think>" in str(content):
                extracted, cleaned = _extract_think_from_content(str(content))
                if extracted:
                    warnings.warn(
                        f"Assistant message has <think> embedded in 'content' instead of "
                        f"using 'reasoning_content' field. Automatically extracted."
                    )
                    content = cleaned
                    # merge with existing reasoning_content if present
                    existing_rc = msg.get("reasoning_content", "")
                    if existing_rc:
                        msg["reasoning_content"] = existing_rc + "\n" + extracted
                    else:
                        msg["reasoning_content"] = extracted

            blocks = []

            # Reasoning content (OpenAI reasoning_content field)
            # controlled by --think: 'all'=keep all, 'last'=keep only last asst, 'none'=strip all
            if think_mode == "all" or (think_mode == "last" and idx == last_assistant_idx):
                reasoning = msg.get("reasoning_content")
                if reasoning and reasoning.strip():
                    blocks.append({"type": "reasoning", "value": reasoning.strip()})

            # Text content (can be empty/None when tool_calls are present)
            text = (content or "").strip()
            if text:
                blocks.append({"type": "text", "value": text})

            # Tool calls (OpenAI tool_calls array → V1 tool_call blocks)
            tool_calls = msg.get("tool_calls", []) or []
            for tc in tool_calls:
                fn = tc.get("function", tc)
                fn_name = fn.get("name", "")
                fn_args = _parse_tool_call_args(fn.get("arguments", {}))

                tc_obj = {"name": fn_name, "arguments": fn_args}
                blocks.append({
                    "type": "tool_call",
                    "value": json.dumps(tc_obj, ensure_ascii=False),
                })

            # If no blocks at all (empty assistant), skip
            if not blocks:
                continue

            messages.append({
                "role": "assistant",
                "content": blocks,
                "loss_weight": 1.0,
            })
            continue

        # ── tool ──
        if role == "tool":
            messages.append({
                "role": "tool",
                "content": [{"type": "text", "value": content}],
                "loss_weight": 0.0,
            })
            continue

        # ── unknown ──
        messages.append({
            "role": "user",
            "content": [{"type": "text", "value": content}],
            "loss_weight": 0.0,
        })

    result: SFTSample = {"messages": messages}

    # Preserve tools field (top-level)
    tools = raw_sample.get("tools")
    if tools:
        if isinstance(tools, list):
            tools = json.dumps(tools, ensure_ascii=False)
        # Ensure it's a JSON string
        elif isinstance(tools, str):
            pass  # already a string
        result["tools"] = tools

    return result


# ── self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = {
        "messages": [
            {"role": "system", "content": "You are a helpful coding agent."},
            {"role": "user", "content": "Find the bug in this code."},
            {"role": "user", "content": "<system-reminder>\nAuto-attached context...\n</system-reminder>"},
            {
                "role": "assistant",
                "reasoning_content": "I need to search for the function definition.",
                "content": "Let me look for it.",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "grep",
                            "arguments": '{"pattern": "def foo", "path": "/src"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_001", "content": "src/utils.py:42: def foo(x): ..."},
            {
                "role": "assistant",
                "reasoning_content": "Found the issue.",
                "content": "The bug is on line 42. Function foo lacks a return statement.",
            },
        ]
    }

    converted = openai_agent_converter(sample)
    print("=== V1 internal format ===")
    print(json.dumps(converted, indent=2, ensure_ascii=False))

    # Round-trip verification
    from llamafactory.v1.core.rendering.format import _to_hf_messages

    hf = _to_hf_messages(converted["messages"])
    print("\n=== HF format (back to OpenAI) ===")
    print(json.dumps(hf, indent=2, ensure_ascii=False))
