#!/usr/bin/env python3
"""
Qwen 多轮 SFT Dataset — 支持两种数据来源：

1. pre-tokenized 模式（Step 2 产出的 Parquet）:
   Parquet 含 input_ids / loss_mask 列时直接读取返回，跳过 tokenize。
   loss_mask 为连续值（已含 token/turn/sqrt 归一化），VeRL 的 sft_loss
   用 masked_sum / mask.sum() 自动适配连续 mask。

2. 运行时 tokenize 模式（普通 messages Parquet）:
   逐消息独立 tokenize + 拼接 + 构造 0/1 loss_mask（与 v1_training_verl
   中旧版 qwen_multiturn_sft_dataset 相同逻辑，保留兼容性）。

两种模式共享同一套 padding/truncation 逻辑（由 VeRL data 配置控制）。
"""

import logging
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class QwenMultiTurnSFTDataset(Dataset):
    """Multi-turn SFT dataset for Qwen models.

    Args:
        parquet_files: Path(s) to Parquet file(s).
        tokenizer: HuggingFace tokenizer（pre-tokenized 模式下仅用于 pad_token_id）。
        config: DictConfig，支持 pad_mode / max_length / truncation / messages_key /
                enable_thinking_key / apply_chat_template_kwargs / loss_mask_key 等。
        processor: Optional multimodal processor.
        max_samples: Limit the number of samples (-1 = use all).
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        self._config = config or {}
        self.pad_mode = self._config.get("pad_mode", "no_padding")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = self._config.get("truncation", "left")
        self.max_length = self._config.get("max_length", 65536)
        self.messages_key = self._config.get("messages_key", "messages")
        self.loss_mask_key = self._config.get("loss_mask_key", "loss_mask")
        self.turn_offsets_key = self._config.get("turn_offsets_key", "turn_offsets")
        # window_length: 滑动窗口的 token 长度阈值。
        #   None / <=0 → 不拆窗，完整数据交给 verl 的 token 级截断（truncation=left 保留尾部）
        #   >0         → 超长对话按轮边界拆滑动窗口，窗口 ≤ window_length
        self.window_length = self._config.get("window_length", None)
        self.window_overlap_turns = self._config.get("window_overlap_turns", 2)
        self.sliding_window = (
            self.window_length is not None and int(self.window_length) > 0
        )
        self.enable_thinking_key = self._config.get("enable_thinking_key", "enable_thinking")
        self.enable_thinking_default = self._config.get("enable_thinking_default", None)
        self.apply_chat_template_kwargs = self._config.get("apply_chat_template_kwargs", {})
        self.max_samples = max_samples
        self.ignore_input_ids_mismatch = self._config.get("ignore_input_ids_mismatch", True)
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        self.parquet_files = parquet_files
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor

        self._download()
        self._read_files_and_process()
        logger.info("QwenMultiTurnSFTDataset ready: %d samples (pre_tokenized=%s)",
                    len(self), self.pre_tokenized)

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            from verl.utils.fs import copy_local_path_from_hdfs
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file, dtype_backend="pyarrow")
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        total = len(self.dataframe)
        print(f"QwenMultiTurnSFTDataset: {total} samples loaded")

        if self.max_samples > 0 and self.max_samples < total:
            seed = self._config.get("seed") if self._config else None
            rng = np.random.default_rng(seed)
            indices = rng.choice(total, size=self.max_samples, replace=False)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"  selected {self.max_samples} random samples out of {total}")

        # ── pre-tokenized 检测：input_ids / loss_mask 列都存在则直通 ──
        self.pre_tokenized = (
            "input_ids" in self.dataframe.columns and self.loss_mask_key in self.dataframe.columns
        )
        if self.pre_tokenized:
            print("QwenMultiTurnSFTDataset: pre-tokenized mode (bypass runtime tokenization)")
            self._build_window_index()
            return

        # ── 运行时 tokenize 模式（滑动窗口不适用，仅 pre-tokenized 支持）──
        self.messages = (
            self.dataframe[self.messages_key]
            .apply(convert_nested_value_to_list_recursive)
            .tolist()
        )
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

    def _build_window_index(self) -> None:
        """pre-tokenized 模式下构建样本索引。

        每条对话:
        - window_length 未设置/<=0 → 1 个条目（完整对话），超长交给
          verl 框架自身的 token 级截断（truncation 配置控制，left = 保留尾部）
        - window_length > 0:
          序列 <= window_length → 1 个条目
          序列 > window_length  → 按轮边界拆窗，保留 window_overlap_turns 轮重叠，
          保证每一轮至少被训练一次（头轮不丢失）
        条目格式: (row_idx, start_token, end_token, sup_start_token)

        sup_start_token: 本窗口内监督起点。与上一窗口重叠的轮次只作上下文
        （loss_mask 置 0，不重复计 loss），监督仅落在上一窗口未覆盖的新轮上，
        保证每轮每个 epoch 严格训练一次。
        """
        self.window_index: list[tuple[int, int, int, int]] = []
        n_windowed = 0
        n_overlong = 0
        for row_idx in range(len(self.dataframe)):
            seq_len = len(self.dataframe.iloc[row_idx]["input_ids"])
            if not self.sliding_window:
                # 不拆窗：完整对话原样给 verl（框架侧 token 截断）
                self.window_index.append((row_idx, 0, seq_len, 0))
                continue

            win_len = int(self.window_length)
            if seq_len <= win_len:
                self.window_index.append((row_idx, 0, seq_len, 0))
                continue

            n_overlong += 1
            offsets = None
            if self.turn_offsets_key in self.dataframe.columns:
                offsets = self.dataframe.iloc[row_idx][self.turn_offsets_key]

            if not offsets:
                # 无轮边界信息：整条给运行时 truncation
                self.window_index.append((row_idx, 0, seq_len, 0))
                continue

            n_windowed += 1
            n_segs = len(offsets)
            cur_start = 0
            prev_end = -1  # 上一窗口已覆盖（且已监督）的最后一段
            while cur_start < n_segs:
                # 找从 cur_start 出发能放下的最大 end（至少含 cur_start 本身）
                end = cur_start
                for e in range(cur_start, n_segs):
                    if offsets[e][1] - offsets[cur_start][0] <= win_len:
                        end = e
                    else:
                        break
                # 与上一窗口重叠的段只作上下文：监督从上一窗口结束后的第一个新段开始
                # （无重叠时 sup_start_seg == cur_start，即全窗监督）
                sup_start_seg = max(cur_start, prev_end + 1)
                sup_start = offsets[sup_start_seg][0]
                self.window_index.append(
                    (row_idx, offsets[cur_start][0], offsets[end][1], sup_start)
                )
                prev_end = end
                if end >= n_segs - 1:
                    break
                cur_start = max(cur_start + 1, end + 1 - self.window_overlap_turns)

        print(f"QwenMultiTurnSFTDataset: {len(self.window_index)} window entries "
              f"({n_overlong} overlong rows, {n_windowed} windowed, "
              f"window_length={self.window_length}, overlap-as-context mask on)")

    def __len__(self):
        if self.pre_tokenized:
            return len(self.window_index)
        return len(self.dataframe)

    # ═══════════════════════════════════════════════════════════════════════════
    # pre-tokenized 直通
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_pre_tokenized(self, item: int) -> dict:
        """读取 Step 2 产出的预 token 化行，应用滑动窗口 + truncation/padding。"""
        row_idx, start_token, end_token, sup_start = self.window_index[item]
        row = self.dataframe.iloc[row_idx]
        input_ids = torch.tensor(list(row["input_ids"])[start_token:end_token], dtype=torch.long)
        attention_mask = torch.tensor(list(row["attention_mask"])[start_token:end_token], dtype=torch.long)
        loss_mask = torch.tensor(list(row[self.loss_mask_key])[start_token:end_token], dtype=torch.float32)
        # 与上一窗口重叠的部分只作上下文：监督区从 sup_start 开始，头部 loss 置 0
        sup_offset = max(0, sup_start - start_token)
        if sup_offset > 0:
            loss_mask[:sup_offset] = 0.0
        # 窗口内 position 从 0 重新编号
        position_ids = torch.arange(end_token - start_token, dtype=torch.long)

        sequence_length = input_ids.shape[0]

        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                pad_len = self.max_length - sequence_length
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=torch.float32)])
                position_ids = F.pad(position_ids, (0, pad_len), value=0)
            elif sequence_length > self.max_length:
                input_ids, attention_mask, loss_mask, position_ids = self._truncate(
                    input_ids, attention_mask, loss_mask, position_ids
                )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            if sequence_length > self.max_length and self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            if sequence_length > self.max_length:
                input_ids, attention_mask, loss_mask, position_ids = self._truncate(
                    input_ids, attention_mask, loss_mask, position_ids
                )
            # 与运行时模式一致：no_padding 不带 attention_mask
            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

    def _truncate(self, input_ids, attention_mask, loss_mask, position_ids):
        if self.truncation == "left":
            input_ids = input_ids[-self.max_length :]
            attention_mask = attention_mask[-self.max_length :]
            loss_mask = loss_mask[-self.max_length :]
            position_ids = position_ids[..., -self.max_length :]
        else:  # right
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            loss_mask = loss_mask[: self.max_length]
            position_ids = position_ids[..., : self.max_length]
        return input_ids, attention_mask, loss_mask, position_ids

    # ═══════════════════════════════════════════════════════════════════════════
    # 运行时 tokenize（兼容普通 messages Parquet，逻辑与旧版一致）
    # ═══════════════════════════════════════════════════════════════════════════

    def _tokenize_single_message(self, msg: dict) -> list[int]:
        try:
            result = normalize_token_ids(
                self.tokenizer.apply_chat_template(
                    [msg], add_generation_prompt=False, tokenize=True
                )
            )
            return result
        except Exception:
            pass

        if msg["role"] == "system":
            content = self._flatten_content(msg.get("content", ""))
            text = f"<|im_start|>system\n{content}<|im_end|>\n"
            return self.tokenizer.encode(text, add_special_tokens=False)

        dummy = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        try:
            dummy_prefix = normalize_token_ids(
                self.tokenizer.apply_chat_template(dummy, add_generation_prompt=False, tokenize=True)
            )
            combined = normalize_token_ids(
                self.tokenizer.apply_chat_template(
                    dummy + [msg], add_generation_prompt=False, tokenize=True
                )
            )
            return combined[len(dummy_prefix):]
        except Exception:
            content = self._flatten_content(msg.get("content", ""))
            text = f"<|im_start|>{msg['role']}\n{content}<|im_end|>\n"
            return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
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

    def _build_inputs(self, item: int):
        row_dict = self.dataframe.iloc[item].to_dict()
        messages = convert_nested_value_to_list_recursive(row_dict[self.messages_key])

        enable_thinking = (
            self.enable_thinking[item]
            if self.enable_thinking is not None
            else self.enable_thinking_default
        )

        all_input_ids = []
        all_loss_mask = []

        for msg in messages:
            msg_tokens = self._tokenize_single_message(msg)

            if msg["role"] == "assistant":
                empty_msg = {"role": "assistant", "content": ""}
                empty_tokens = self._tokenize_single_message(empty_msg)
                gen_len = len(empty_tokens)

                if gen_len < len(msg_tokens):
                    all_input_ids.extend(msg_tokens)
                    all_loss_mask.extend([0] * gen_len + [1] * (len(msg_tokens) - gen_len))
                else:
                    all_input_ids.extend(msg_tokens)
                    all_loss_mask.extend([1] * len(msg_tokens))
            else:
                all_input_ids.extend(msg_tokens)
                all_loss_mask.extend([0] * len(msg_tokens))

        input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        loss_mask = torch.tensor(all_loss_mask, dtype=torch.float32)
        return input_ids, loss_mask

    def __getitem__(self, item):
        if self.pre_tokenized:
            return self._get_pre_tokenized(item)

        input_ids, loss_mask = self._build_inputs(item)
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        sequence_length = len(input_ids)

        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                pad_len = self.max_length - sequence_length
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=torch.float32)])
                position_ids = F.pad(position_ids, (0, pad_len), value=0)
                return {
                    "input_ids": input_ids,
                    "attention_mask": torch.cat([
                        torch.ones(sequence_length, dtype=torch.long),
                        torch.zeros(pad_len, dtype=torch.long),
                    ]),
                    "position_ids": position_ids,
                    "loss_mask": loss_mask,
                }
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                    position_ids = position_ids[..., -self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                    position_ids = position_ids[..., : self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                return {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                    "position_ids": position_ids,
                    "loss_mask": loss_mask,
                }
            else:
                return {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones(sequence_length, dtype=torch.long),
                    "position_ids": position_ids,
                    "loss_mask": loss_mask,
                }
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            if sequence_length > self.max_length and self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            if sequence_length > self.max_length:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[..., : self.max_length]
            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")
