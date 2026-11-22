"""
PyTorch Dataset for FaDe training on ConflictBank contradiction pairs.

Each item yields the activation-path input (evidence in sampled formats), the
anchoring-path input (all-text evidence), and the cached teacher targets.

Stochastic format augmentation is performed on-the-fly in __getitem__,
so each epoch naturally sees different format combinations.
"""

from typing import Callable, Dict, List, Optional
import logging
import random
import torch
from torch.utils.data import Dataset

from .teacher_cache import TeacherCache
from .conflictbank_processor import ALL_FORMATS

logger = logging.getLogger(__name__)

IGNORE_LABEL = -100
FORMAT_TO_ID = {fmt: i for i, fmt in enumerate(ALL_FORMATS)}


def tokenize_prompt_response(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Tokenize a (prompt, response) pair and return ids, mask, prompt_len."""
    prompt_msg = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msg, tokenize=False, add_generation_prompt=True,
    )
    full_msg = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(full_msg, tokenize=False)

    prompt_tok = tokenizer(
        prompt_text, return_tensors="pt", truncation=True,
        max_length=max_length, add_special_tokens=False,
    )
    prompt_len = prompt_tok["attention_mask"].sum().item()

    full_tok = tokenizer(
        full_text, return_tensors="pt", truncation=True,
        max_length=max_length, add_special_tokens=False,
    )
    return {
        "input_ids": full_tok["input_ids"].squeeze(0),
        "attention_mask": full_tok["attention_mask"].squeeze(0),
        "prompt_len": prompt_len,
    }


class ConflictBankDataset(Dataset):
    """Dataset for FaDe training on ConflictBank contradiction pairs.

    Each sample is a contradiction pair with two evidence pieces. Stochastic
    format augmentation assigns random formats to each evidence piece.
    """

    def __init__(
        self,
        conflict_pairs: list,
        tokenizer: Callable,
        teacher_cache: TeacherCache,
        max_length: int = 4096,
        format_mode: str = "random_4way",
        anchor_ratio: float = 0.0,
        compute_segment_masks: bool = False,
    ):
        self.pairs = list(conflict_pairs)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_mode = format_mode
        self.anchor_ratio = anchor_ratio
        self.teacher_cache = teacher_cache
        self.compute_segment_masks = compute_segment_masks
        self._validate_teacher_cache_length()

    def _validate_teacher_cache_length(self):
        if len(self.teacher_cache) < len(self.pairs):
            raise ValueError(
                f"Teacher cache has {len(self.teacher_cache)} entries but "
                f"dataset has {len(self.pairs)} pairs."
            )
        if len(self.teacher_cache) > len(self.pairs):
            logger.warning(
                "Teacher cache has %d entries but dataset uses %d pairs; "
                "only the matching prefix will be used.",
                len(self.teacher_cache), len(self.pairs),
            )
        cached_ids = self.teacher_cache.meta.get("sample_ids", [])
        if cached_ids:
            expected_ids = [
                str(pair.get("id", i)) for i, pair in enumerate(self.pairs)
            ]
            if cached_ids[:len(expected_ids)] != expected_ids:
                raise ValueError(
                    "Teacher cache sample IDs do not match the current "
                    "ConflictBank dataset prefix. Rebuild the cache with the "
                    "same data path, sample cap, and symmetrization flags."
                )

    def __len__(self):
        return len(self.pairs)

    def _tokenize(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        return tokenize_prompt_response(self.tokenizer, prompt, response, self.max_length)

    def _compute_segment_masks(
        self, prompt: str, input_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Find token-level masks for Source A and Source B evidence segments."""
        seq_len = input_ids.size(0)
        mask_a = torch.zeros(seq_len, dtype=torch.bool)
        mask_b = torch.zeros(seq_len, dtype=torch.bool)

        sa_marker = "Source A:\n"
        sb_marker = "\n\nSource B:\n"
        q_marker = "\n\nQuestion:"

        sa_start_char = prompt.find(sa_marker)
        sb_start_char = prompt.find(sb_marker)
        q_start_char = prompt.find(q_marker)

        if sa_start_char < 0 or sb_start_char < 0 or q_start_char < 0:
            return {"seg_a": mask_a, "seg_b": mask_b}

        ev_a_start = sa_start_char + len(sa_marker)
        ev_a_end = sb_start_char
        ev_b_start = sb_start_char + len(sb_marker)
        ev_b_end = q_start_char

        full_msg = [{"role": "user", "content": prompt}]
        full_text = self.tokenizer.apply_chat_template(
            full_msg, tokenize=False, add_generation_prompt=True,
        )
        chat_offset = full_text.find(prompt)
        if chat_offset < 0:
            return {"seg_a": mask_a, "seg_b": mask_b}

        ev_a_start += chat_offset
        ev_a_end += chat_offset
        ev_b_start += chat_offset
        ev_b_end += chat_offset

        tok_out = self.tokenizer(
            full_text, return_offsets_mapping=True,
            truncation=True, max_length=self.max_length,
            add_special_tokens=False,
        )
        offsets = tok_out.get("offset_mapping", [])
        if not offsets:
            return {"seg_a": mask_a, "seg_b": mask_b}

        for i, (s, e) in enumerate(offsets):
            if i >= seq_len:
                break
            if s >= ev_a_start and e <= ev_a_end and e > s:
                mask_a[i] = True
            if s >= ev_b_start and e <= ev_b_end and e > s:
                mask_b[i] = True

        return {"seg_a": mask_a, "seg_b": mask_b}

    def __getitem__(self, idx):
        from .conflictbank_processor import (
            build_conflict_prompt,
            build_conflict_prompt_all_text,
            _sample_format_pair,
        )

        pair = self.pairs[idx]

        answer = pair.get("answer_a", pair.get("answer", ""))
        is_anchor = random.random() < self.anchor_ratio

        teacher_data = build_conflict_prompt_all_text(pair)
        teacher_prompt = teacher_data["prompt"]

        order_swapped = False
        if is_anchor:
            student_prompt = teacher_prompt
            assigned_formats = ("text", "text")
        else:
            fmt_a, fmt_b = _sample_format_pair(self.format_mode)
            student_data = build_conflict_prompt(
                pair, format_a=fmt_a, format_b=fmt_b, randomize_order=True,
            )
            student_prompt = student_data["prompt"]
            assigned_formats = (fmt_a, fmt_b)
            order_swapped = bool(student_data.get("swapped", False))

        student_act = self._tokenize(student_prompt, answer)
        student_anc = self._tokenize(teacher_prompt, answer)

        if self.compute_segment_masks:
            student_act["segment_masks"] = self._compute_segment_masks(
                student_prompt, student_act["input_ids"],
            )

        # Format labels follow the rendered Source A / Source B order.
        fmt_first, fmt_second = assigned_formats
        if order_swapped:
            fmt_first, fmt_second = fmt_second, fmt_first

        return {
            "student_act": student_act,
            "student_anc": student_anc,
            "is_anchor": is_anchor,
            "format_labels": torch.tensor(
                [
                    FORMAT_TO_ID.get(fmt_first, IGNORE_LABEL),
                    FORMAT_TO_ID.get(fmt_second, IGNORE_LABEL),
                ],
                dtype=torch.long,
            ),
            "meta": {
                "pair_id": pair.get("id", ""),
                "conflict_type": pair.get("conflict_type", ""),
                "assigned_formats": list(assigned_formats),
            },
            "teacher_cache": self.teacher_cache[idx],
        }


def collate_fade(batch: List[Dict]) -> Dict:
    """Pad student inputs and cached teacher targets into a batch."""
    result = {}

    for key in ["student_act", "student_anc"]:
        input_ids = [item[key]["input_ids"] for item in batch]
        attn_masks = [item[key]["attention_mask"] for item in batch]
        prompt_lens = [item[key]["prompt_len"] for item in batch]

        max_len = max(ids.size(0) for ids in input_ids)
        padded_ids = torch.zeros(len(batch), max_len, dtype=input_ids[0].dtype)
        padded_mask = torch.zeros(len(batch), max_len, dtype=attn_masks[0].dtype)

        for i, (ids, mask) in enumerate(zip(input_ids, attn_masks)):
            padded_ids[i, :ids.size(0)] = ids
            padded_mask[i, :mask.size(0)] = mask

        result[key] = {
            "input_ids": padded_ids,
            "attention_mask": padded_mask,
            "prompt_lens": prompt_lens,
        }

    def _pad_cache_target(index_key, logprob_key, resp_len_key):
        max_resp = max(
            item["teacher_cache"][index_key].size(0) for item in batch
        )
        K = batch[0]["teacher_cache"][index_key].size(-1)

        padded_indices = torch.zeros(
            len(batch), max_resp, K, dtype=torch.int32,
        )
        padded_logprobs = torch.full((len(batch), max_resp, K), -20.0)
        resp_lens = []

        for i, item in enumerate(batch):
            tc = item["teacher_cache"]
            rlen = int(tc.get(resp_len_key, tc[index_key].size(0)))
            copy_len = min(rlen, tc[index_key].size(0), max_resp)
            padded_indices[i, :copy_len] = tc[index_key][:copy_len]
            padded_logprobs[i, :copy_len] = tc[logprob_key][:copy_len]
            resp_lens.append(copy_len)

        return padded_indices, padded_logprobs, resp_lens

    padded_indices, padded_logprobs, resp_lens = _pad_cache_target(
        "topk_indices", "topk_logprobs", "resp_len",
    )
    orig_idx, orig_lp, orig_lens = _pad_cache_target(
        "original_topk_indices",
        "original_topk_logprobs",
        "original_resp_len",
    )
    result["teacher_cache"] = {
        "topk_indices": padded_indices,
        "topk_logprobs": padded_logprobs,
        "resp_lens": resp_lens,
        "original_topk_indices": orig_idx,
        "original_topk_logprobs": orig_lp,
        "original_resp_lens": orig_lens,
    }

    result["is_anchor"] = [item["is_anchor"] for item in batch]
    result["format_labels"] = torch.stack(
        [item["format_labels"] for item in batch],
    )

    if "segment_masks" in batch[0]["student_act"]:
        max_len = result["student_act"]["input_ids"].size(1)
        seg_a_list, seg_b_list = [], []
        for item in batch:
            sm = item["student_act"]["segment_masks"]
            sa = sm["seg_a"]
            sb = sm["seg_b"]
            pad_len = max_len - sa.size(0)
            if pad_len > 0:
                sa = torch.cat([sa, torch.zeros(pad_len, dtype=torch.bool)])
                sb = torch.cat([sb, torch.zeros(pad_len, dtype=torch.bool)])
            seg_a_list.append(sa)
            seg_b_list.append(sb)
        result["segment_masks"] = {
            "seg_a": torch.stack(seg_a_list),
            "seg_b": torch.stack(seg_b_list),
        }

    return result
