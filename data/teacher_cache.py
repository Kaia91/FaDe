"""
Offline Teacher Logits Caching.

Removes the teacher model from GPU during training by pre-computing and
storing the teacher's top-K targets for each training sample.

For each sample the frozen teacher is run on the same content under the
format assignments in `teacher_prompts_backdoor` (the 7 evaluation format
pairs, text-text first). The response distributions are averaged into a
format-marginalized target, which is mixed with the original all-text target
with a weight set by the mean cross-format JS divergence (adaptive
marginalization). The original all-text target is stored as well.

Cache format per sample:
  {
    "sample_id": str,
    "topk_indices": Tensor[L_resp, K],            # format-marginalized target
    "topk_logprobs": Tensor[L_resp, K],           # log-probabilities (renormalized)
    "original_topk_indices": Tensor[L_resp, K],   # original all-text target
    "original_topk_logprobs": Tensor[L_resp, K],
    "prompt_len": int,
    "total_len": int,
    ...
  }

Stored as .pt files, one per shard.
"""

import os
import json
import logging
import math
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

try:
    import flash_attn  # noqa: F401
    _ATTN_IMPL = "flash_attention_2"
except ImportError:
    _ATTN_IMPL = "sdpa"


def _tokenize_teacher_sample(tokenizer, prompt, answer, max_length):
    """Tokenize a (prompt, answer) pair; return tensors + prompt_len."""
    prompt_msg = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msg, tokenize=False, add_generation_prompt=True,
    )
    full_msg = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    full_text = tokenizer.apply_chat_template(full_msg, tokenize=False)

    prompt_tok = tokenizer(
        prompt_text, return_tensors="pt", truncation=True,
        max_length=max_length, add_special_tokens=False,
    )
    plen = prompt_tok["attention_mask"].sum().item()

    full_tok = tokenizer(
        full_text, return_tensors="pt", truncation=True,
        max_length=max_length, add_special_tokens=False,
    )
    return {
        "input_ids": full_tok["input_ids"].squeeze(0),
        "attention_mask": full_tok["attention_mask"].squeeze(0),
        "prompt_len": plen,
    }


def _mean_js_divergence(
    ref_log_probs: torch.Tensor,
    alt_log_probs: torch.Tensor,
) -> Optional[float]:
    """Mean Jensen-Shannon divergence over aligned response positions."""
    r = min(ref_log_probs.size(0), alt_log_probs.size(0))
    if r <= 0:
        return None

    ref = ref_log_probs[:r].float()
    alt = alt_log_probs[:r].float()
    mix = torch.logaddexp(ref, alt) - math.log(2.0)
    js = 0.5 * (
        (ref.exp() * (ref - mix)).sum(dim=-1)
        + (alt.exp() * (alt - mix)).sum(dim=-1)
    )
    return float(js.mean().item())


def _adaptive_backdoor_weight(
    disagreement: float,
    low: float,
    high: float,
    min_weight: float,
    max_weight: float,
) -> float:
    """Map format disagreement to a mixing weight on the marginalized target."""
    if high <= low:
        raise ValueError("adaptive_backdoor_high must be greater than low")
    if not 0.0 <= min_weight <= max_weight <= 1.0:
        raise ValueError(
            "adaptive backdoor weights must satisfy 0 <= min <= max <= 1"
        )
    frac = (float(disagreement) - low) / (high - low)
    frac = max(0.0, min(1.0, frac))
    return float(min_weight + frac * (max_weight - min_weight))


def _mix_logprob_distributions(
    original_log_probs: torch.Tensor,
    backdoor_log_probs: torch.Tensor,
    backdoor_weight: float,
) -> torch.Tensor:
    """Arithmetic mixture of original and marginalized full-vocab log-probs."""
    r = min(original_log_probs.size(0), backdoor_log_probs.size(0))
    if r <= 0:
        return backdoor_log_probs

    alpha = max(0.0, min(1.0, float(backdoor_weight)))
    if alpha <= 0.0:
        return original_log_probs[:r].clone()
    if alpha >= 1.0:
        return backdoor_log_probs[:r].clone()

    mixed = torch.logaddexp(
        original_log_probs[:r] + math.log1p(-alpha),
        backdoor_log_probs[:r] + math.log(alpha),
    )
    return mixed - torch.logsumexp(mixed, dim=-1, keepdim=True)


def cache_teacher_logits(
    model_name: str,
    samples: List[Dict],
    cache_dir: str,
    top_k: int = 256,
    max_length: int = 4096,
    batch_size: int = 4,
    shard_size: int = 500,
    bf16: bool = True,
    model_cache_dir: Optional[str] = None,
    overwrite: bool = False,
    adaptive_backdoor_low: float = 0.01,
    adaptive_backdoor_high: float = 0.05,
    adaptive_backdoor_min_weight: float = 0.0,
    adaptive_backdoor_max_weight: float = 1.0,
    extra_meta: Optional[Dict] = None,
) -> str:
    """Pre-compute and cache the teacher's top-K targets.

    The teacher sees each sample's prompts + gold answer. For each response
    token position, we store the top-K token indices and their renormalized
    log-probabilities.

    Args:
        model_name: HF model name or local path for the teacher model.
        samples: List of dicts with keys 'teacher_prompt' (all-text),
                 'teacher_prompts_backdoor' (same content under each format
                 assignment), 'teacher_format_pairs_backdoor' and 'answer'.
        cache_dir: Directory to write cache shards.
        top_k: Number of top tokens to cache per position.
        max_length: Max sequence length for tokenization.
        batch_size: Batch size for teacher forward passes.
        shard_size: Samples per shard file.
        bf16: Use bfloat16 for the teacher model.
        model_cache_dir: Optional HF cache dir for model weights.
        overwrite: If True, remove existing cache shard files in cache_dir
                   before writing. If False, existing shards are treated as
                   a configuration error to prevent silent cache mixing.
        adaptive_backdoor_low/high: JS divergence thresholds (tau_low,
                   tau_high) for the linear original->marginalized mixing
                   schedule.
        adaptive_backdoor_min/max_weight: Clamp range (alpha_min, alpha_max)
                   for the weight on the marginalized target.
        extra_meta: Additional fields to record in cache_meta.json.

    Returns:
        Path to the cache directory containing shard files.
    """
    _adaptive_backdoor_weight(
        adaptive_backdoor_low,
        adaptive_backdoor_low,
        adaptive_backdoor_high,
        adaptive_backdoor_min_weight,
        adaptive_backdoor_max_weight,
    )

    os.makedirs(cache_dir, exist_ok=True)
    existing_shards = sorted(
        f for f in os.listdir(cache_dir) if f.endswith(".pt")
    )
    if existing_shards:
        if not overwrite:
            raise RuntimeError(
                f"Teacher cache directory already contains {len(existing_shards)} "
                f"shard files: {cache_dir}. Use --overwrite_teacher_cache or "
                "choose a new --teacher_cache_dir to avoid mixing stale logits."
            )
        for sf in existing_shards:
            os.remove(os.path.join(cache_dir, sf))
        meta_path = os.path.join(cache_dir, "cache_meta.json")
        if os.path.exists(meta_path):
            os.remove(meta_path)
        logger.info(f"Overwriting existing teacher cache at {cache_dir}")

    cache_meta = {
        "num_samples": len(samples),
        "top_k": top_k,
        "max_length": max_length,
        "bf16": bf16,
        "causal_backdoor": True,
        "backdoor_format_mode": "eval_pairs",
        "cache_original_teacher_target": True,
        "adaptive_backdoor_marginalization": True,
        "adaptive_backdoor_low": adaptive_backdoor_low,
        "adaptive_backdoor_high": adaptive_backdoor_high,
        "adaptive_backdoor_min_weight": adaptive_backdoor_min_weight,
        "adaptive_backdoor_max_weight": adaptive_backdoor_max_weight,
        "adaptive_backdoor_metric": "mean_response_js_full_vocab",
        "backdoor_format_pairs": (
            samples[0].get("teacher_format_pairs_backdoor", []) if samples else []
        ),
        "sample_ids": [
            str(item.get("pair_id", item.get("id", item.get("sample_id", i))))
            for i, item in enumerate(samples)
        ],
    }
    if extra_meta:
        cache_meta.update(extra_meta)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=model_cache_dir, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if bf16 else torch.float16
    logger.info(f"Loading teacher model from {model_name} for caching...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=model_cache_dir, torch_dtype=dtype,
        attn_implementation=_ATTN_IMPL, trust_remote_code=True,
    ).cuda().eval()

    for p in model.parameters():
        p.requires_grad = False

    all_cached = []
    pbar = tqdm(range(0, len(samples), batch_size), desc="Caching teacher logits")

    for batch_start in pbar:
        batch = samples[batch_start: batch_start + batch_size]

        # --- Primary all-text forward ---
        tok_primary = [
            _tokenize_teacher_sample(
                tokenizer, item["teacher_prompt"],
                item.get("answer", ""), max_length,
            )
            for item in batch
        ]
        prompt_lens = [t["prompt_len"] for t in tok_primary]
        input_ids_list = [t["input_ids"] for t in tok_primary]
        attn_mask_list = [t["attention_mask"] for t in tok_primary]

        max_len = max(ids.size(0) for ids in input_ids_list)
        padded_ids = torch.zeros(len(batch), max_len, dtype=input_ids_list[0].dtype)
        padded_mask = torch.zeros(len(batch), max_len, dtype=attn_mask_list[0].dtype)
        for i, (ids, mask) in enumerate(zip(input_ids_list, attn_mask_list)):
            padded_ids[i, :ids.size(0)] = ids
            padded_mask[i, :mask.size(0)] = mask
        padded_ids = padded_ids.cuda()
        padded_mask = padded_mask.cuda()

        with torch.no_grad():
            logits = model(
                input_ids=padded_ids, attention_mask=padded_mask,
            ).logits

        # Only response-token logits are ever cached or mixed. Computing
        # full-sequence full-vocab log_softmax can take several GB at long
        # context lengths and can OOM under multi-format caching.
        primary_resp_log_probs = []
        for i in range(len(batch)):
            plen = prompt_lens[i]
            seq_len = attn_mask_list[i].sum().item()
            resp_start = max(plen - 1, 0)
            resp_end = seq_len
            if resp_end <= resp_start:
                primary_resp_log_probs.append(None)
            else:
                primary_resp_log_probs.append(
                    F.log_softmax(
                        logits[i, resp_start:resp_end, :].float(), dim=-1,
                    )
                )
        del logits

        # --- Format interventions: same content, each format assignment ---
        variant_prompts = []
        variant_pairs = []
        for item in batch:
            prompts = item.get("teacher_prompts_backdoor")
            if not prompts:
                prompts = [item["teacher_prompt"]]
            variant_prompts.append(prompts)
            variant_pairs.append(item.get("teacher_format_pairs_backdoor", []))

        backdoor_log_probs = [None for _ in batch]
        backdoor_counts = [0 for _ in batch]
        backdoor_disagreement_sums = [0.0 for _ in batch]
        backdoor_disagreement_counts = [0 for _ in batch]

        def _add_backdoor_slice(sample_idx: int, resp_lp: torch.Tensor) -> None:
            # Running arithmetic mean of the per-format distributions.
            count = backdoor_counts[sample_idx]
            if count == 0:
                backdoor_log_probs[sample_idx] = resp_lp.clone()
            else:
                prev = backdoor_log_probs[sample_idx]
                r = min(prev.size(0), resp_lp.size(0))
                if r <= 0:
                    return
                w_prev = math.log(count / (count + 1.0))
                w_new = math.log(1.0 / (count + 1.0))
                mixed = torch.logaddexp(
                    prev[:r] + w_prev,
                    resp_lp[:r] + w_new,
                )
                # `mixed` is already a normalized arithmetic mixture of
                # probabilities, but renormalize for numerical drift.
                mixed = mixed - torch.logsumexp(mixed, dim=-1, keepdim=True)
                backdoor_log_probs[sample_idx] = mixed
            backdoor_counts[sample_idx] = count + 1

        def _add_adaptive_disagreement(
            sample_idx: int,
            resp_lp: torch.Tensor,
        ) -> None:
            ref_lp = primary_resp_log_probs[sample_idx]
            if ref_lp is None:
                return
            js = _mean_js_divergence(ref_lp, resp_lp)
            if js is None:
                return
            backdoor_disagreement_sums[sample_idx] += js
            backdoor_disagreement_counts[sample_idx] += 1

        # Reuse the primary all-text forward when it is the first
        # format assignment (text-text).
        variant_start = []
        for i, prompts in enumerate(variant_prompts):
            start_idx = 0
            if prompts and prompts[0] == batch[i]["teacher_prompt"]:
                plen = prompt_lens[i]
                seq_len = attn_mask_list[i].sum().item()
                resp_start = max(plen - 1, 0)
                resp_end = seq_len
                if resp_end > resp_start and primary_resp_log_probs[i] is not None:
                    _add_backdoor_slice(i, primary_resp_log_probs[i])
                start_idx = 1
            variant_start.append(start_idx)

        max_variants = max(len(p) for p in variant_prompts)
        for variant_idx in range(max_variants):
            active = [
                (i, variant_idx, prompts[variant_idx])
                for i, prompts in enumerate(variant_prompts)
                if variant_idx >= variant_start[i] and variant_idx < len(prompts)
            ]
            if not active:
                continue

            tok_variant = [
                _tokenize_teacher_sample(
                    tokenizer, prompt, batch[i].get("answer", ""), max_length,
                )
                for i, _, prompt in active
            ]
            ids_list = [t["input_ids"] for t in tok_variant]
            mask_list = [t["attention_mask"] for t in tok_variant]
            plens_v = [t["prompt_len"] for t in tok_variant]

            max_len_v = max(ids.size(0) for ids in ids_list)
            padded_ids_v = torch.zeros(
                len(active), max_len_v, dtype=ids_list[0].dtype,
            )
            padded_mask_v = torch.zeros(
                len(active), max_len_v, dtype=mask_list[0].dtype,
            )
            for j, (ids, mask) in enumerate(zip(ids_list, mask_list)):
                padded_ids_v[j, :ids.size(0)] = ids
                padded_mask_v[j, :mask.size(0)] = mask
            padded_ids_v = padded_ids_v.cuda()
            padded_mask_v = padded_mask_v.cuda()

            with torch.no_grad():
                logits_v = model(
                    input_ids=padded_ids_v,
                    attention_mask=padded_mask_v,
                ).logits

            for j, (sample_idx, _, _) in enumerate(active):
                plen_v = plens_v[j]
                seq_len_v = mask_list[j].sum().item()
                resp_start_v = max(plen_v - 1, 0)
                resp_end_v = seq_len_v
                if resp_end_v <= resp_start_v:
                    continue
                resp_lp = F.log_softmax(
                    logits_v[j, resp_start_v:resp_end_v, :].float(),
                    dim=-1,
                )
                _add_backdoor_slice(sample_idx, resp_lp)
                _add_adaptive_disagreement(sample_idx, resp_lp)
            del logits_v

        for i in range(len(batch)):
            plen = prompt_lens[i]
            seq_len = attn_mask_list[i].sum().item()
            resp_start = max(plen - 1, 0)
            resp_end = seq_len
            if resp_end <= resp_start:
                continue

            resp_log_probs = (
                backdoor_log_probs[i]
                if backdoor_log_probs[i] is not None
                else primary_resp_log_probs[i]
            )
            if resp_log_probs is None:
                continue

            # Adaptive marginalization: weight on the marginalized target
            # grows with the mean cross-format disagreement.
            adaptive_disagreement = None
            adaptive_weight = None
            if (primary_resp_log_probs[i] is not None
                    and backdoor_log_probs[i] is not None):
                denom = max(backdoor_disagreement_counts[i], 1)
                adaptive_disagreement = (
                    backdoor_disagreement_sums[i] / denom
                )
                adaptive_weight = _adaptive_backdoor_weight(
                    adaptive_disagreement,
                    adaptive_backdoor_low,
                    adaptive_backdoor_high,
                    adaptive_backdoor_min_weight,
                    adaptive_backdoor_max_weight,
                )
                resp_log_probs = _mix_logprob_distributions(
                    primary_resp_log_probs[i],
                    backdoor_log_probs[i],
                    adaptive_weight,
                )

            topk_logprobs, topk_indices = resp_log_probs.topk(top_k, dim=-1)
            topk_logprobs = topk_logprobs - torch.logsumexp(topk_logprobs, dim=-1, keepdim=True)

            entry = {
                "sample_id": str(batch_start + i),
                "topk_indices": topk_indices.cpu().to(torch.int32),
                "topk_logprobs": topk_logprobs.cpu().to(torch.float16),
                "prompt_len": plen,
                "total_len": seq_len,
                "resp_len": resp_log_probs.size(0),
                "backdoor_num_variants": int(backdoor_counts[i]),
                "adaptive_backdoor_disagreement": float(adaptive_disagreement or 0.0),
                "adaptive_backdoor_weight": float(adaptive_weight or 0.0),
            }
            if variant_pairs[i]:
                entry["backdoor_format_pairs"] = variant_pairs[i]

            original_resp_log_probs = primary_resp_log_probs[i]
            if original_resp_log_probs is not None:
                orig_lp, orig_idx = original_resp_log_probs.topk(
                    top_k, dim=-1,
                )
                orig_lp = orig_lp - torch.logsumexp(
                    orig_lp, dim=-1, keepdim=True,
                )
                entry["original_topk_indices"] = (
                    orig_idx.cpu().to(torch.int32)
                )
                entry["original_topk_logprobs"] = (
                    orig_lp.cpu().to(torch.float16)
                )
                entry["original_prompt_len"] = plen
                entry["original_total_len"] = seq_len
                entry["original_resp_len"] = original_resp_log_probs.size(0)

            all_cached.append(entry)

        if len(all_cached) >= shard_size:
            shard_idx = len([f for f in os.listdir(cache_dir) if f.endswith(".pt")])
            shard_path = os.path.join(cache_dir, f"shard_{shard_idx:04d}.pt")
            torch.save(all_cached, shard_path)
            logger.info(f"Saved shard {shard_path} ({len(all_cached)} samples)")
            all_cached = []

    if all_cached:
        shard_idx = len([f for f in os.listdir(cache_dir) if f.endswith(".pt")])
        shard_path = os.path.join(cache_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(all_cached, shard_path)
        logger.info(f"Saved final shard {shard_path} ({len(all_cached)} samples)")

    with open(os.path.join(cache_dir, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(cache_meta, f, ensure_ascii=False, indent=2)

    del model
    torch.cuda.empty_cache()
    logger.info(f"Teacher cache complete: {cache_dir}")
    return cache_dir


class TeacherCache:
    """In-memory cache loader for pre-computed teacher logits."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.data = []
        self.meta = {}
        self._load()

    def _load(self):
        meta_path = os.path.join(self.cache_dir, "cache_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            logger.warning(
                f"Teacher cache metadata not found at {meta_path}; "
                "cache/data compatibility cannot be fully validated."
            )

        shard_files = sorted(
            f for f in os.listdir(self.cache_dir) if f.endswith(".pt")
        )
        for sf in shard_files:
            shard = torch.load(
                os.path.join(self.cache_dir, sf), map_location="cpu",
            )
            self.data.extend(shard)
        logger.info(f"Loaded {len(self.data)} cached teacher entries from {self.cache_dir}")
        expected = self.meta.get("num_samples")
        if expected is not None and expected != len(self.data):
            raise ValueError(
                f"Teacher cache metadata expects {expected} entries but "
                f"loaded {len(self.data)} from {self.cache_dir}."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        out = {
            "topk_indices": item["topk_indices"],
            "topk_logprobs": item["topk_logprobs"].float(),
            "prompt_len": item["prompt_len"],
            "total_len": item["total_len"],
            "resp_len": item.get("resp_len", item["topk_indices"].size(0)),
        }
        if "original_topk_indices" in item:
            out["original_topk_indices"] = item["original_topk_indices"]
            out["original_topk_logprobs"] = item["original_topk_logprobs"].float()
            out["original_prompt_len"] = item.get(
                "original_prompt_len", item["prompt_len"],
            )
            out["original_total_len"] = item.get(
                "original_total_len", item["total_len"],
            )
            out["original_resp_len"] = item.get(
                "original_resp_len", item["original_topk_indices"].size(0),
            )
        return out
