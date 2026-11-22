"""
FaDe trainer.

The frozen teacher's top-K targets are pre-computed offline
(data/teacher_cache.py); only the LoRA student is on the GPU during training.
Stochastic format augmentation gives each epoch different format combinations.

The training loop:
  - For each batch, the student gets two forward passes:
    (a) Activation path: student sees the evidence in sampled formats and is
        matched to the format-marginalized teacher target
    (b) Anchoring path:  student sees the all-text input and is matched to
        the original all-text teacher target
  - Loss = L_act + lambda_anc * L_anc + lambda_gold * L_gold
           + lambda_bal * L_bal + lambda_adv * L_adv  (see losses/loss_design.py)
"""

import os
import logging
import shutil
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

from losses.loss_design import FaDeLoss, IGNORE_INDEX
from data.dataset import ConflictBankDataset, collate_fade
from data.teacher_cache import TeacherCache

logger = logging.getLogger(__name__)

try:
    import flash_attn  # noqa: F401
    _ATTN_IMPL = "flash_attention_2"
except ImportError:
    _ATTN_IMPL = "sdpa"
    logger.warning("flash_attn not found, falling back to SDPA attention.")


class FaDeTrainer:
    """Trainer for FaDe cross-format knowledge distillation."""

    def __init__(
        self,
        model_name: str,
        teacher_cache_dir: str,
        cache_dir: Optional[str] = None,
        learning_rate: float = 3e-6,
        anchor_coef: float = 0.0,
        top_k: int = 256,
        max_length: int = 4096,
        bf16: bool = True,
        gradient_checkpointing: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        balance_coef: float = 0.1,
        gold_ce_coef: float = 0.2,
        format_adv_coef: float = 0.05,
    ):
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.use_balance = balance_coef > 0
        self.use_gold_ce = gold_ce_coef > 0
        self.use_format_adv = format_adv_coef > 0

        logger.info(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if bf16 else torch.float16

        # --- Student: base model + LoRA ---
        logger.info(f"Loading student model from {model_name}")
        base_student = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=dtype,
            attn_implementation=_ATTN_IMPL, trust_remote_code=True,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        self.student_model = get_peft_model(base_student, lora_config).cuda()
        trainable, total = self.student_model.get_nb_trainable_parameters()
        logger.info(
            f"Student LoRA: trainable={trainable:,} / total={total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

        if gradient_checkpointing:
            self.student_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # --- Teacher: cached targets ---
        logger.info(f"Using cached teacher logits from {teacher_cache_dir}")
        self.teacher_cache = TeacherCache(teacher_cache_dir)
        has_original = (
            len(self.teacher_cache.data) > 0
            and "original_topk_indices" in self.teacher_cache.data[0]
        )
        if not has_original:
            raise ValueError(
                "Teacher cache has no original all-text targets; rebuild it "
                "with this code's cache stage."
            )

        # --- Loss function ---
        logger.info(
            f"Loss config: top_k={top_k}, anchor_coef={anchor_coef}, "
            f"balance_coef={balance_coef}, gold_ce_coef={gold_ce_coef}, "
            f"format_adv_coef={format_adv_coef}"
        )
        self.loss_fn = FaDeLoss(
            anchor_coef=anchor_coef,
            top_k=top_k,
            balance_coef=balance_coef,
            gold_ce_coef=gold_ce_coef,
            format_adv_coef=format_adv_coef,
            format_adv_hidden_size=getattr(base_student.config, "hidden_size", None),
        ).cuda()

    # ------------------------------------------------------------------
    # Response-level target alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cached_teacher_aligned(
        teacher_cache_batch: dict,
        student_prompt_lens: List[int],
        student_mask: torch.Tensor,
        device: torch.device,
        prefix: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build aligned labels + padded cached tensors for loss computation.

        Returns (topk_indices, topk_logprobs, labels) all padded to (B, L, K).
        L is the max response length in the batch.
        """
        idx_key = f"{prefix}topk_indices"
        lp_key = f"{prefix}topk_logprobs"
        lens_key = f"{prefix}resp_lens"
        if idx_key not in teacher_cache_batch or lp_key not in teacher_cache_batch:
            raise ValueError(
                f"Teacher cache batch is missing '{idx_key}' / '{lp_key}'."
            )

        topk_idx = teacher_cache_batch[idx_key].to(device)
        topk_lp = teacher_cache_batch[lp_key].to(device)
        resp_lens = teacher_cache_batch[lens_key]

        B = topk_idx.size(0)
        s_seq = student_mask.sum(dim=1).long()
        s_resp = [s_seq[i].item() - student_prompt_lens[i] for i in range(B)]

        aligned = [max(min(rl, sr), 0) for rl, sr in zip(resp_lens, s_resp)]
        max_r = max(aligned) if aligned else 0

        if max_r <= 0:
            K = topk_idx.size(-1)
            dummy_idx = torch.zeros(B, 2, K, dtype=torch.int32, device=device)
            dummy_lp = torch.full((B, 2, K), -20.0, device=device)
            labels = torch.full((B, 2), IGNORE_INDEX, device=device, dtype=torch.long)
            return dummy_idx, dummy_lp, labels

        L = max_r + 1

        labels = torch.full((B, L), IGNORE_INDEX, device=device, dtype=torch.long)
        for i in range(B):
            r = aligned[i]
            if r > 0:
                labels[i, 1:r + 1] = 0

        padded_idx = torch.zeros(B, L, topk_idx.size(-1), dtype=topk_idx.dtype, device=device)
        padded_lp = torch.full((B, L, topk_lp.size(-1)), -20.0, device=device, dtype=topk_lp.dtype)

        for i in range(B):
            r = aligned[i]
            if r > 0:
                padded_idx[i, :r + 1] = topk_idx[i, :r + 1]
                padded_lp[i, :r + 1] = topk_lp[i, :r + 1]

        return padded_idx, padded_lp, labels

    @staticmethod
    def _mix_cached_teacher_aligned(
        primary_idx: torch.Tensor,
        primary_lp: torch.Tensor,
        primary_labels: torch.Tensor,
        alternate_idx: torch.Tensor,
        alternate_lp: torch.Tensor,
        alternate_labels: torch.Tensor,
        use_alternate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select per-sample cached targets after response alignment.

        Dual-teacher routing uses the format-marginalized targets for
        activation samples, but the original all-text teacher targets for
        anchor samples.
        """
        B = primary_idx.size(0)
        K = primary_idx.size(-1)
        L = max(primary_idx.size(1), alternate_idx.size(1))
        device = primary_idx.device

        mixed_idx = torch.zeros(
            B, L, K, dtype=primary_idx.dtype, device=device,
        )
        mixed_lp = torch.full(
            (B, L, K), -20.0, dtype=primary_lp.dtype, device=device,
        )
        mixed_labels = torch.full(
            (B, L), IGNORE_INDEX, dtype=primary_labels.dtype, device=device,
        )

        use_alternate = use_alternate.to(device=device, dtype=torch.bool)
        for i in range(B):
            if use_alternate[i]:
                src_idx, src_lp, src_labels = (
                    alternate_idx[i], alternate_lp[i], alternate_labels[i],
                )
            else:
                src_idx, src_lp, src_labels = (
                    primary_idx[i], primary_lp[i], primary_labels[i],
                )
            n = src_idx.size(0)
            mixed_idx[i, :n] = src_idx
            mixed_lp[i, :n] = src_lp
            mixed_labels[i, :src_labels.size(0)] = src_labels

        return mixed_idx, mixed_lp, mixed_labels

    @staticmethod
    def _build_response_target_ids(
        input_ids: torch.Tensor,
        prompt_lens: List[int],
        mask: torch.Tensor,
        target_len: int,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build next-token gold labels aligned with extracted response logits.

        `_extract_student_response` starts at `prompt_len - 1`, so position 0
        predicts the first response token stored at `input_ids[prompt_len]`.
        We keep index 0 ignored and fill target positions 1..N to match the
        KL mask convention used throughout this trainer.
        """
        B = input_ids.size(0)
        device = input_ids.device
        targets = torch.full(
            (B, target_len), IGNORE_INDEX, device=device, dtype=torch.long,
        )
        seq_lens = mask.sum(dim=1).long()
        for i in range(B):
            resp_len = max(seq_lens[i].item() - prompt_lens[i], 0)
            n = min(resp_len, max(target_len - 1, 0))
            if n <= 0:
                continue
            start = prompt_lens[i]
            targets[i, 1:n + 1] = input_ids[i, start:start + n]

        if labels is not None:
            targets = targets.masked_fill(labels.eq(IGNORE_INDEX), IGNORE_INDEX)
        return targets

    @staticmethod
    def _extract_student_response(
        student_logits: torch.Tensor,
        prompt_lens: List[int],
        mask: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """Extract student response logits and pad to target_len, preserving gradients."""
        B, _, V = student_logits.shape
        s_seq = mask.sum(dim=1).long()

        slices = []
        for i in range(B):
            start = max(prompt_lens[i] - 1, 0)
            end = min(start + target_len, s_seq[i].item())
            sl = student_logits[i, start:end]
            if sl.size(0) < target_len:
                sl = F.pad(sl, (0, 0, 0, target_len - sl.size(0)))
            slices.append(sl)
        return torch.stack(slices, dim=0)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, save_path, optimizer, scheduler, epoch, step, global_step):
        ckpt_dir = os.path.join(save_path, "checkpoints", f"step_{global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.student_model.save_pretrained(ckpt_dir)
        state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "step": step, "global_step": global_step,
        }
        loss_state = self.loss_fn.state_dict()
        if loss_state:
            state["loss_fn"] = loss_state
        torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))
        logger.info(f"Checkpoint saved: {ckpt_dir}")

        ckpts_root = os.path.join(save_path, "checkpoints")
        existing = sorted(
            [d for d in os.listdir(ckpts_root)
             if d.startswith("step_") and os.path.isdir(os.path.join(ckpts_root, d))],
            key=lambda x: int(x.split("_")[1]),
        )
        while len(existing) > 2:
            checkpoint_to_remove = os.path.join(ckpts_root, existing.pop(0))
            shutil.rmtree(checkpoint_to_remove, ignore_errors=True)

    def _load_latest_checkpoint(self, save_path):
        ckpts_root = os.path.join(save_path, "checkpoints")
        if not os.path.isdir(ckpts_root):
            return None
        dirs = [d for d in os.listdir(ckpts_root)
                if d.startswith("step_") and os.path.isdir(os.path.join(ckpts_root, d))]
        if not dirs:
            return None
        return os.path.join(ckpts_root, max(dirs, key=lambda x: int(x.split("_")[1])))

    def _build_optimizer(self):
        groups = [{
            "params": [
                p for p in self.student_model.parameters() if p.requires_grad
            ],
        }]
        aux_params = [p for p in self.loss_fn.parameters() if p.requires_grad]
        if aux_params:
            groups.append({"params": aux_params})
            logger.info(
                "Optimizer includes %d format-adversary parameters",
                sum(p.numel() for p in aux_params),
            )
        return torch.optim.AdamW(
            groups, lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=0.01,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        conflict_pairs,
        num_epochs: int = 2,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 16,
        warmup_ratio: float = 0.05,
        save_path: str = "./outputs/fade",
        save_steps: int = 50,
        format_mode: str = "random_4way",
        anchor_ratio: float = 0.0,
        resume: bool = True,
    ):
        """Train the LoRA student on ConflictBank pairs."""
        # Segment masks are needed for the hidden-state balance and
        # format-adversarial losses.
        need_seg_masks = self.use_balance or self.use_format_adv
        dataset = ConflictBankDataset(
            conflict_pairs=conflict_pairs,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            format_mode=format_mode,
            anchor_ratio=anchor_ratio,
            teacher_cache=self.teacher_cache,
            compute_segment_masks=need_seg_masks,
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fade,
        )

        optimizer = self._build_optimizer()
        total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=max(total_steps, 1),
        )

        start_epoch, start_step, global_step = 0, 0, 0
        if resume:
            ckpt_dir = self._load_latest_checkpoint(save_path)
            if ckpt_dir:
                state_path = os.path.join(ckpt_dir, "training_state.pt")
                if os.path.exists(state_path):
                    logger.info(f"Resuming from {ckpt_dir}")
                    self.student_model.load_adapter(ckpt_dir, adapter_name="default")
                    state = torch.load(state_path, map_location="cpu")
                    if "loss_fn" in state:
                        self.loss_fn.load_state_dict(state["loss_fn"], strict=False)
                    optimizer.load_state_dict(state["optimizer"])
                    scheduler.load_state_dict(state["scheduler"])
                    start_epoch = state["epoch"]
                    start_step = state["step"] + 1
                    global_step = state["global_step"]

        self.student_model.train()

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [ConflictBank]")

            for step, batch in enumerate(pbar):
                if epoch == start_epoch and step < start_step:
                    continue

                # --- Student forward: activation path ---
                act_ids = batch["student_act"]["input_ids"].cuda()
                act_mask = batch["student_act"]["attention_mask"].cuda()
                act_plens = batch["student_act"]["prompt_lens"]

                has_seg = "segment_masks" in batch
                need_hs = (self.use_balance or self.use_format_adv) and has_seg

                act_out = self.student_model(
                    input_ids=act_ids, attention_mask=act_mask,
                    output_hidden_states=need_hs,
                )
                student_act_logits = act_out.logits
                hidden_states_act = act_out.hidden_states if need_hs else None
                del act_out

                # --- Student forward: anchoring path ---
                anc_ids = batch["student_anc"]["input_ids"].cuda()
                anc_mask = batch["student_anc"]["attention_mask"].cuda()
                anc_plens = batch["student_anc"]["prompt_lens"]

                student_anc_logits = self.student_model(
                    input_ids=anc_ids, attention_mask=anc_mask,
                ).logits

                aux_kwargs = {}
                if need_hs:
                    aux_kwargs["hidden_states_act"] = hidden_states_act
                    aux_kwargs["seg_a_mask"] = batch["segment_masks"]["seg_a"].cuda()
                    aux_kwargs["seg_b_mask"] = batch["segment_masks"]["seg_b"].cuda()
                if self.use_format_adv and "format_labels" in batch:
                    aux_kwargs["format_labels"] = batch["format_labels"].cuda()

                # --- Teacher targets ---
                # Activation path: format-marginalized target, except for
                # all-text anchor samples, which use the original target.
                tc_idx_act, tc_lp_act, labels_act = self._build_cached_teacher_aligned(
                    batch["teacher_cache"], act_plens, act_mask, act_ids.device,
                )
                orig_idx_act, orig_lp_act, orig_labels_act = (
                    self._build_cached_teacher_aligned(
                        batch["teacher_cache"],
                        act_plens,
                        act_mask,
                        act_ids.device,
                        prefix="original_",
                    )
                )
                use_original_act = torch.tensor(
                    batch["is_anchor"],
                    device=act_ids.device,
                    dtype=torch.bool,
                )
                tc_idx_act, tc_lp_act, labels_act = (
                    self._mix_cached_teacher_aligned(
                        tc_idx_act,
                        tc_lp_act,
                        labels_act,
                        orig_idx_act,
                        orig_lp_act,
                        orig_labels_act,
                        use_original_act,
                    )
                )

                # Anchoring path: original all-text target.
                tc_idx_anc, tc_lp_anc, labels_anc = self._build_cached_teacher_aligned(
                    batch["teacher_cache"],
                    anc_plens,
                    anc_mask,
                    anc_ids.device,
                    prefix="original_",
                )

                L_act = labels_act.size(1)
                aligned_s_act = self._extract_student_response(
                    student_act_logits, act_plens, act_mask, L_act,
                )
                aligned_s_anc = self._extract_student_response(
                    student_anc_logits, anc_plens, anc_mask, labels_anc.size(1),
                )

                loss_kwargs = {
                    "teacher_topk_indices_act": tc_idx_act,
                    "teacher_topk_logprobs_act": tc_lp_act,
                    "teacher_topk_indices_anc": tc_idx_anc,
                    "teacher_topk_logprobs_anc": tc_lp_anc,
                }
                if self.use_gold_ce:
                    loss_kwargs["target_ids_act"] = self._build_response_target_ids(
                        act_ids, act_plens, act_mask, labels_act.size(1), labels_act,
                    )
                    loss_kwargs["target_ids_anc"] = self._build_response_target_ids(
                        anc_ids, anc_plens, anc_mask, labels_anc.size(1), labels_anc,
                    )

                losses = self.loss_fn(
                    student_act_logits=aligned_s_act,
                    student_anc_logits=aligned_s_anc,
                    labels_act=labels_act,
                    labels_anc=labels_anc,
                    **loss_kwargs,
                    **aux_kwargs,
                )

                loss = losses["loss"] / gradient_accumulation_steps
                if loss.requires_grad:
                    loss.backward()
                else:
                    logger.warning(f"Skipping backward at step {step}: no grad_fn")

                epoch_loss += losses["loss"].item()
                n_batches += 1

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group["params"]],
                        1.0,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % save_steps == 0:
                        self._save_checkpoint(
                            save_path, optimizer, scheduler, epoch, step, global_step,
                        )

                postfix = {
                    "loss": f"{losses['loss'].item():.4f}",
                    "act": f"{losses['loss_act'].item():.4f}",
                    "anc": f"{losses['loss_anc'].item():.4f}",
                    "gs": global_step,
                }
                if self.use_balance:
                    postfix["bal"] = f"{losses['loss_bal'].item():.4f}"
                if self.use_gold_ce:
                    postfix["ce"] = f"{losses['loss_gold_ce'].item():.4f}"
                if self.use_format_adv:
                    postfix["fadv"] = f"{losses['loss_fmt_adv'].item():.4f}"
                pbar.set_postfix(postfix)

            avg = epoch_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch+1} avg loss: {avg:.4f}")
            self._save_checkpoint(
                save_path, optimizer, scheduler, epoch, len(dataloader) - 1, global_step,
            )
            start_step = 0

        os.makedirs(save_path, exist_ok=True)
        lora_path = os.path.join(save_path, "lora_adapter")
        self.student_model.save_pretrained(lora_path)
        logger.info(f"LoRA adapter saved to {lora_path}")

        merged = self.student_model.merge_and_unload()
        merged.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Merged model saved to {save_path}")
