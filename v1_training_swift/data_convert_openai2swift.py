#!/usr/bin/env python3
"""
将 OpenAI 格式的 agent 训练数据转换为 MSWIFT agent_template 兼容格式。

转换内容：
1. reasoning_content 字段 → assistant content 中的 <think>...</think> 标签
2. assistant 消息上的 tool_calls 数组 → 独立的 tool_call 角色消息
3. tool 角色规范化（保留 tool 角色，去掉 tool_call_id）
4. 连续 user 消息合并（<system-reminder> 注入合并到前一条 user 消息）
5. 构造 top-level tools 字段（工具定义）
6. system 消息去重（同一会话中可能多次出现相同 system 消息）

用法:
    python data_convert_openai2swift.py

配置（修改 main() 中的变量）:
    - INPUT_DIR: 输入目录
    - OUTPUT_DIR: 输出目录
"""

import json
import os
import glob
import re
from collections import OrderedDict
from copy import deepcopy


# ============================================================
# 工具定义 — 根据实际数据中出现的工具构建
# ============================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the local filesystem. Returns the file content with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "The line number to start reading from"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The number of lines to read"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory or file path to search in"
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": "Output mode"
                    },
                    "-n": {
                        "type": "boolean",
                        "description": "Show line numbers"
                    },
                    "-i": {
                        "type": "boolean",
                        "description": "Case-insensitive search"
                    },
                    "-r": {
                        "type": "boolean",
                        "description": "Recursive search"
                    }
                },
                "required": ["pattern", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of what this command does"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in milliseconds"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Performs exact string replacement in a file. old_string must match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to modify"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The text to replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences instead of just the first one",
                        "default": False
                    }
                },
                "required": ["file_path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file to the local filesystem. Overwrites if one exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match"
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in"
                    }
                },
                "required": ["pattern", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory (optionally recursive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list files recursively",
                        "default": False
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_tool",
            "description": "Launch a subagent to handle complex, multi-step tasks autonomously.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subagent_type": {
                        "type": "string",
                        "description": "The type of subagent to launch"
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Detailed description of what the subagent should do"
                    }
                },
                "required": ["subagent_type", "task_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo_create",
            "description": "Create a new task in the task list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "A brief title for the task"
                    },
                    "description": {
                        "type": "string",
                        "description": "What needs to be done"
                    }
                },
                "required": ["subject", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo_modify",
            "description": "Modify or update a task in the task list (status, assignee, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the task to modify"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "deleted"],
                        "description": "New status for the task"
                    }
                },
                "required": ["task_id"]
            }
        }
    },
]


# ============================================================
# 转换逻辑
# ============================================================

def is_system_reminder(content: str) -> bool:
    """判断是否为 <system-reminder> 自动注入消息（训练噪声）。"""
    return bool(content and '<system-reminder>' in content)


def parse_tool_call_arguments(args):
    """解析 tool_call 的 arguments 字段，确保返回 dict。"""
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


def convert_messages(messages: list) -> tuple[list, dict]:
    """
    将 OpenAI 格式的 messages 转换为 MSWIFT agent 格式。

    主要转换:
    1. reasoning_content → content 中的 <think> 标签
    2. tool_calls 数组 → 独立 tool_call 角色消息
    3. assistant 的 content（当有 tool_calls 时）→ 保留为前导 assistant 消息
    4. 连续 user 消息合并

    Returns:
        (new_messages, stats) — stats 包含各类丢弃/转换计数
    """
    new_messages = []
    stats = {
        'system_reminder_discarded': 0,   # 丢弃的 <system-reminder> 消息
        'user_merged': 0,                  # 合并的连续 user 消息
        'reasoning_merged': 0,             # reasoning_content 合并次数
        'tool_calls_split': 0,             # tool_calls 拆分为 tool_call 的次数
    }

    for msg in messages:
        role = msg.get('role', '')

        # ============================================================
        # system 消息: 直接保留
        # ============================================================
        if role == 'system':
            new_messages.append({
                'role': 'system',
                'content': msg['content']
            })
            continue

        # ============================================================
        # user 消息: 连续 user 合并 + 过滤 system-reminder
        # ============================================================
        if role == 'user':
            content = msg.get('content', '') or ''

            # <system-reminder> 注入：始终丢弃。
            # 这些是 harness 自动注入的上下文，不是真正的 user turn，
            # 保留它们会破坏 tool → assistant 的交替模式，
            # 导致 MSWIFT encoding 时触发:
            #   assert response_role in {'assistant'}
            if is_system_reminder(content):
                stats['system_reminder_discarded'] += 1
                continue

            # 连续 user 消息合并
            if new_messages and new_messages[-1]['role'] == 'user':
                new_messages[-1]['content'] += '\n\n' + content
                stats['user_merged'] += 1
            else:
                new_messages.append({
                    'role': 'user',
                    'content': content
                })
            continue

        # ============================================================
        # assistant 消息: 处理 reasoning_content + tool_calls
        # ============================================================
        if role == 'assistant':
            has_rc = 'reasoning_content' in msg and msg['reasoning_content']
            has_tc = 'tool_calls' in msg and msg['tool_calls']
            has_content = bool(msg.get('content'))

            # 构建 content：合并 reasoning_content + 原始 content
            content_parts = []
            if has_rc and msg['reasoning_content']:
                reasoning = msg['reasoning_content'].strip()
                if reasoning:
                    content_parts.append(f'<think>\n{reasoning}\n</think>')
                    stats['reasoning_merged'] += 1
            if has_content and msg.get('content'):
                content_parts.append(msg['content'].strip())

            combined_content = '\n\n'.join(content_parts) if content_parts else ''

            # 添加 assistant 消息（如果有内容，或者没有 tool_calls 时至少有个空壳）
            if has_tc:
                # 有 tool_calls: 只有有实质内容时才添加前导 assistant 消息
                # 如果既没有 reasoning 也没有 content，完全不生成 assistant 消息
                if content_parts:
                    new_messages.append({
                        'role': 'assistant',
                        'content': combined_content
                    })

                # 将 tool_calls 数组拆分为独立的 tool_call 角色消息
                for tc in msg['tool_calls']:
                    fn = tc.get('function', tc)
                    fn_name = fn.get('name', '')
                    fn_args = parse_tool_call_arguments(fn.get('arguments', {}))

                    new_messages.append({
                        'role': 'tool_call',
                        'content': json.dumps({
                            'name': fn_name,
                            'arguments': fn_args
                        }, ensure_ascii=False)
                    })
                    stats['tool_calls_split'] += 1
            else:
                # 无 tool_calls: 普通 assistant 回复（可能是最终答案）
                new_messages.append({
                    'role': 'assistant',
                    'content': combined_content
                })
            continue

        # ============================================================
        # tool 消息: 规范化（去掉 tool_call_id，保留 json 字符串内容）
        # ============================================================
        if role == 'tool':
            content = msg.get('content', '') or ''

            new_messages.append({
                'role': 'tool',
                'content': content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            })
            continue

        # ============================================================
        # tool_call / tool_response: 如果数据中已有这些 role，直接保留
        # ============================================================
        if role in ('tool_call', 'tool_response'):
            content = msg.get('content', '') or ''
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            new_messages.append({
                'role': role,
                'content': content
            })
            continue

        # ============================================================
        # 未知 role: 保留原样
        # ============================================================
        new_messages.append(msg)

    return new_messages, stats


def collect_used_tool_names(item: dict) -> set:
    """从一个 item 中收集所有出现过的工具名。"""
    tool_names = set()
    for msg in item.get('messages', []):
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            for tc in msg['tool_calls']:
                fn = tc.get('function', tc)
                name = fn.get('name', '')
                if name:
                    tool_names.add(name)
        # 转换后的 tool_call 消息
        if msg.get('role') == 'tool_call':
            try:
                obj = json.loads(msg['content']) if isinstance(msg['content'], str) else msg['content']
                if obj.get('name'):
                    tool_names.add(obj['name'])
            except (json.JSONDecodeError, TypeError):
                pass
    return tool_names


def build_tools_field(tool_names: set) -> str:
    """根据工具名集合构建 tools 字段（JSON 字符串）。"""
    if not tool_names:
        # 返回所有工具的完整定义
        return json.dumps(TOOL_DEFINITIONS, ensure_ascii=False)

    tools_to_use = []
    for tool_def in TOOL_DEFINITIONS:
        fn_name = tool_def['function']['name']
        if fn_name in tool_names:
            tools_to_use.append(tool_def)

    return json.dumps(tools_to_use, ensure_ascii=False)


def convert_item(item: dict, build_tools: bool = True) -> tuple[dict, dict]:
    """
    转换单条训练数据。

    Args:
        item: 原始数据项（含 messages 字段）
        build_tools: 是否自动构建 tools 字段

    Returns:
        (converted_item, stats) — stats 包含各类转换计数
    """
    result = {}

    # 保留已有的顶层字段（如 images, rejected_response 等）
    for key, value in item.items():
        if key == 'messages':
            continue
        if key == 'tools':
            # 已有 tools 字段则保留
            result['tools'] = value
            continue
        result[key] = value

    # 转换 messages
    result['messages'], convert_stats = convert_messages(item.get('messages', []))

    # 构建 tools 字段（如果还没有）
    if build_tools and 'tools' not in result:
        tool_names = collect_used_tool_names(item)
        result['tools'] = build_tools_field(tool_names)

    return result, convert_stats


def deduplicate_system_messages(item: dict) -> dict:
    """
    移除连续重复的 system 消息（同一 system prompt 重复出现）。
    只在 role 为 assistant 后的 system 才可能是噪声重复。
    """
    messages = item.get('messages', [])
    cleaned = []
    for i, msg in enumerate(messages):
        if msg['role'] == 'system' and cleaned:
            # 检查是否与前一条 system 完全相同
            prev = cleaned[-1]
            if prev['role'] == 'system' and prev.get('content') == msg.get('content'):
                continue
        cleaned.append(msg)
    item['messages'] = cleaned
    return item


def cleanup_empty_assistant_messages(item: dict) -> dict:
    """移除 content 为空的 assistant 消息。"""
    messages = item.get('messages', [])
    cleaned = []
    for msg in messages:
        if msg['role'] == 'assistant' and not msg.get('content', '').strip():
            continue
        cleaned.append(msg)
    item['messages'] = cleaned
    return item


def check_item_quality(item: dict, item_idx: int = 0) -> dict:
    """
    检查转换后数据项的质量，特别是 MSWIFT 编码兼容性。

    模拟 _swift_prepare_inputs 后的状态，检测:
    1. 不完整对话：预处理后消息数以 'tool' 结尾（奇偶配对会丢弃最后一条）
    2. 无训练信号：去除 system 后消息数 < 2（没有 assistant 回复）
    3. 角色交替违规：配对后 query_role 不在 {user, tool} 或 response_role 不是 assistant

    Returns:
        {'ok': bool, 'issues': [str]}
    """
    issues = []
    msgs = item.get('messages', [])

    # 去掉 system 消息（MSWIFT 在 StdTemplateInputs 构造时 pop 掉）
    if msgs and msgs[0].get('role') == 'system':
        msgs = msgs[1:]

    if len(msgs) < 2:
        issues.append(f'Item {item_idx}: only {len(msgs)} messages after removing system — no assistant response to train on')
        return {'ok': False, 'issues': issues}

    # 模拟 _preprocess_tool_call: tool_call → assistant
    simulated = []
    i = 0
    while i < len(msgs):
        role = msgs[i]['role']
        if role == 'tool_call':
            while i + 1 < len(msgs) and msgs[i + 1]['role'] == 'tool_call':
                i += 1
            simulated.append({'role': 'assistant'})
        else:
            simulated.append({'role': role})
        i += 1

    # 合并连续同角色
    merged = []
    j = 0
    while j < len(simulated):
        role = simulated[j]['role']
        if role == 'assistant' and merged and merged[-1]['role'] == 'assistant':
            j += 1
            continue
        elif role == 'tool' and merged and merged[-1]['role'] == 'assistant':
            while j + 1 < len(simulated) and simulated[j + 1]['role'] == 'tool':
                j += 1
            merged.append({'role': 'tool'})
        elif role == 'user' and merged and merged[-1]['role'] == 'user':
            j += 1
            continue
        else:
            merged.append({'role': role})
        j += 1

    # 检查角色交替约束
    n_round = len(merged) // 2
    for k, (q_msg, r_msg) in enumerate(zip(merged[::2], merged[1::2])):
        q_role = q_msg['role']
        r_role = r_msg['role']
        if q_role not in ('user', 'tool'):
            issues.append(f'Item {item_idx}, round {k}: bad query_role={q_role} (expected user or tool)')
        if r_role != 'assistant':
            issues.append(f'Item {item_idx}, round {k}: bad response_role={r_role} (expected assistant)')

    # 检查是否以 tool 结尾（不完整对话）
    if len(merged) % 2 != 0:
        last_role = merged[-1]['role']
        if last_role == 'tool':
            issues.append(
                f'Item {item_idx}: ends with "tool" role — conversation truncated; '
                f'last tool response has no assistant follow-up'
            )
        else:
            issues.append(
                f'Item {item_idx}: odd message count ({len(merged)}), last role={last_role}'
            )

    return {'ok': len(issues) == 0, 'issues': issues}


def convert_file(input_path: str, output_path: str, filter_incomplete: bool = False) -> dict:
    """
    转换单个 JSON 文件。

    Returns:
        {'input_items': int, 'output_items': int, 'errors': int}
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {input_path}, got {type(data)}")

    stats = {
        'input_items': len(data),
        'output_items': 0,
        'errors': 0,
        'filtered_no_training': 0,
        'filtered_incomplete': 0,
        'system_reminder_discarded': 0,
        'user_merged': 0,
        'reasoning_merged': 0,
        'tool_calls_split': 0,
    }
    converted = []

    for i, item in enumerate(data):
        try:
            conv, item_stats = convert_item(item)
            conv = deduplicate_system_messages(conv)
            conv = cleanup_empty_assistant_messages(conv)

            # 质量过滤
            if filter_incomplete:
                quality = check_item_quality(conv, i)
                if not quality['ok']:
                    # 区分两类问题
                    for issue in quality['issues']:
                        if 'no assistant response' in issue:
                            stats['filtered_no_training'] += 1
                            break
                        elif 'ends with "tool"' in issue:
                            stats['filtered_incomplete'] += 1
                            break
                    continue  # 跳过此 item

            converted.append(conv)
            stats['output_items'] += 1
            for key in ('system_reminder_discarded', 'user_merged',
                        'reasoning_merged', 'tool_calls_split'):
                stats[key] += item_stats[key]
        except Exception as e:
            print(f"  [ERROR] Item {i} in {input_path}: {e}")
            stats['errors'] += 1

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    return stats


def validate_item(item: dict, index: int = 0) -> list:
    """
    验证转换后的数据项是否符合 MSWIFT 格式要求。

    Returns:
        错误列表（空列表表示验证通过）
    """
    errors = []

    if 'messages' not in item:
        errors.append(f"Item {index}: missing 'messages' field")
        return errors

    msgs = item['messages']
    if not msgs:
        errors.append(f"Item {index}: empty messages")
        return errors

    # 检查 role 合法性
    valid_roles = {'system', 'user', 'assistant', 'tool_call', 'tool_response', 'tool'}
    for j, msg in enumerate(msgs):
        role = msg.get('role', '')
        if role not in valid_roles:
            errors.append(f"Item {index}, msg {j}: invalid role '{role}'")
        if 'content' not in msg:
            errors.append(f"Item {index}, msg {j}: missing 'content'")

        # tool_call 的 content 应该能解析为 JSON
        if role == 'tool_call':
            try:
                obj = json.loads(msg['content']) if isinstance(msg['content'], str) else msg['content']
                if 'name' not in obj:
                    errors.append(f"Item {index}, msg {j}: tool_call missing 'name'")
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                errors.append(f"Item {index}, msg {j}: tool_call content not valid JSON: {e}")

    # 检查 tools 字段格式
    if 'tools' in item:
        try:
            tools = json.loads(item['tools']) if isinstance(item['tools'], str) else item['tools']
            if not isinstance(tools, list):
                errors.append(f"Item {index}: 'tools' should be a list, got {type(tools)}")
        except (json.JSONDecodeError, TypeError) as e:
            errors.append(f"Item {index}: 'tools' not valid JSON: {e}")

    return errors


# ============================================================
# 主流程
# ============================================================

def main():
    # ---- 配置 ----
    INPUT_DIR = '/data2/cyr/development/coding-sft/data_openai'
    OUTPUT_DIR = '/data2/cyr/development/coding-sft/data_swift'
    BUILD_TOOLS = True             # 是否自动构建 tools 字段
    FILTER_INCOMPLETE = True       # 是否过滤质量不合格的数据（无训练信号/截断对话）
    VALIDATE_OUTPUT = True         # 是否验证输出
    VALIDATE_SAMPLE_SIZE = 10      # 验证时抽检的样本数

    # ----------------------------------------------------------

    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, '**/*.json'), recursive=True))

    if not input_files:
        print(f"[ERROR] No JSON files found in {INPUT_DIR}")
        return

    print(f"Found {len(input_files)} JSON files in {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Build tools: {BUILD_TOOLS}")
    print(f"Filter incomplete: {FILTER_INCOMPLETE}")
    print()

    total_stats = {
        'input_items': 0, 'output_items': 0, 'errors': 0, 'files': 0,
        'filtered_no_training': 0, 'filtered_incomplete': 0,
        'system_reminder_discarded': 0, 'user_merged': 0,
        'reasoning_merged': 0, 'tool_calls_split': 0,
    }
    all_converted = []

    for input_path in input_files:
        # 保持相同的目录结构
        rel_path = os.path.relpath(input_path, INPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, rel_path)

        print(f"Converting: {rel_path} ... ", end='', flush=True)
        try:
            stats = convert_file(input_path, output_path, FILTER_INCOMPLETE)
            filtered = stats['filtered_no_training'] + stats['filtered_incomplete']
            msg = f"OK ({stats['input_items']} → {stats['output_items']} items"
            if filtered:
                msg += f", {filtered} filtered"
            if stats['errors']:
                msg += f", {stats['errors']} errors"
            msg += ")"
            print(msg)
            for key in total_stats:
                if key in stats:
                    total_stats[key] += stats[key]
            total_stats['files'] += 1

            # 收集部分样本用于验证
            if VALIDATE_OUTPUT and len(all_converted) < VALIDATE_SAMPLE_SIZE:
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_converted.extend(data[:VALIDATE_SAMPLE_SIZE - len(all_converted)])

        except Exception as e:
            print(f"FAILED: {e}")
            total_stats['errors'] += 1

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Files processed:   {total_stats['files']}/{len(input_files)}")
    print(f"  Items:             {total_stats['input_items']} → {total_stats['output_items']}")
    if sum([total_stats['filtered_no_training'], total_stats['filtered_incomplete']]):
        print(f"    Filtered (no training signal):  {total_stats['filtered_no_training']:>4}")
        print(f"    Filtered (truncated/incomplete):{total_stats['filtered_incomplete']:>4}")
    if total_stats['errors']:
        print(f"  Errors:            {total_stats['errors']}")
    print()
    print("  Conversions applied:")
    print(f"    <system-reminder> discarded:  {total_stats['system_reminder_discarded']:>6}")
    print(f"    reasoning_content → <think>:  {total_stats['reasoning_merged']:>6}")
    print(f"    tool_calls → tool_call msgs:  {total_stats['tool_calls_split']:>6}")
    print(f"    consecutive users merged:     {total_stats['user_merged']:>6}")

    # ---- 质量检查（全部输出文件） ----
    QUALITY_CHECK = True
    if QUALITY_CHECK:
        print()
        print("=" * 60)
        if FILTER_INCOMPLETE:
            print("Quality Check — verifying filtered output is clean:")
        else:
            print("Quality Check (simulating MSWIFT _swift_prepare_inputs):")
        quality_issues = {'incomplete': [], 'no_training': [], 'constraint': []}
        output_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, '**/*.json'), recursive=True))
        for ofp in output_files:
            with open(ofp, 'r', encoding='utf-8') as f:
                odata = json.load(f)
            for item_idx, oitem in enumerate(odata):
                result = check_item_quality(oitem, item_idx)
                if result['ok']:
                    continue
                for issue in result['issues']:
                    if 'only' in issue and 'no assistant response' in issue:
                        quality_issues['no_training'].append(f"{os.path.basename(ofp)}:{issue}")
                    elif 'ends with "tool"' in issue:
                        quality_issues['incomplete'].append(f"{os.path.basename(ofp)}:{issue}")
                    else:
                        quality_issues['constraint'].append(f"{os.path.basename(ofp)}:{issue}")

        print(f"  No training signal (< 2 msgs): {len(quality_issues['no_training'])} items")
        print(f"  Incomplete (ends with tool):    {len(quality_issues['incomplete'])} items")
        print(f"  Constraint violations:          {len(quality_issues['constraint'])} items")

        if not FILTER_INCOMPLETE:
            if quality_issues['no_training']:
                print(f"\n  No-training-signal items (no assistant response to learn from):")
                for iss in quality_issues['no_training'][:5]:
                    print(f"    - {iss}")
                if len(quality_issues['no_training']) > 5:
                    print(f"    ... and {len(quality_issues['no_training']) - 5} more")

            if quality_issues['incomplete']:
                print(f"\n  Incomplete items (truncated conversation — last assistant step missing):")
                for iss in quality_issues['incomplete'][:5]:
                    print(f"    - {iss}")
                if len(quality_issues['incomplete']) > 5:
                    print(f"    ... and {len(quality_issues['incomplete']) - 5} more")
                print(f"\n  Note: These are NOT format errors. MSWIFT silently drops the unpaired")
                print(f"  last message, so training will run, but the model won't learn what")
                print(f"  to do after the final tool results. Consider filtering these items")
        else:
            if quality_issues['no_training'] or quality_issues['incomplete']:
                print(f"\n  [WARNING] Some items still fail quality check — this should not happen")
                print(f"  with FILTER_INCOMPLETE=True. Check the filtering logic.")
            else:
                print(f"\n  [OK] All output items pass quality check.")

    # ---- 验证 ----
    if VALIDATE_OUTPUT and all_converted:
        print()
        print("=" * 60)
        print(f"Validating {min(VALIDATE_SAMPLE_SIZE, len(all_converted))} sample items...")
        all_errors = []
        for i, item in enumerate(all_converted[:VALIDATE_SAMPLE_SIZE]):
            errors = validate_item(item, i)
            if errors:
                all_errors.extend(errors)

        if all_errors:
            print(f"  [FAIL] {len(all_errors)} validation errors:")
            for e in all_errors[:20]:
                print(f"    - {e}")
            if len(all_errors) > 20:
                print(f"    ... and {len(all_errors) - 20} more")
        else:
            print("  [OK] All samples passed validation")

    # ---- 打印一条转换样本 ----
    if all_converted:
        print()
        print("=" * 60)
        print("Sample converted item (first 10 messages):")
        sample = all_converted[0]
        print(f"  tools: {str(sample.get('tools', ''))[:150]}...")
        print(f"  messages ({len(sample['messages'])} total, showing first 10):")
        for j, msg in enumerate(sample['messages'][:10]):
            role = msg.get('role', '?')
            content = str(msg.get('content', ''))
            if len(content) > 150:
                content = content[:150] + '...'
            print(f"    [{j}] {role}: {content}")
        if len(sample['messages']) > 10:
            print(f"    ... ({len(sample['messages']) - 10} more messages)")

    print()
    print("Done!")


if __name__ == '__main__':
    main()
