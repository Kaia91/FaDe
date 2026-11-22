"""
FaDe entry point.

Two stages:
  Stage 0: Cache the frozen teacher's top-K targets offline
  Stage 1: Train the FaDe student (cross-format KL distillation)

The default arguments are the settings of our FaDe runs.

Usage:
    python train.py --model_name <MODEL_NAME_OR_PATH> --stage all \
        --data_path ./dataset/conflictbank/conflict_train.jsonl \
        --output_dir ./outputs/fade
"""

import argparse
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="FaDe training")

    p.add_argument("--stage", type=str, default="all",
                    choices=["cache", "train", "all"],
                    help="Pipeline stage to run")

    # Model
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--bf16", action="store_true", default=False,
                    help="Use bfloat16 precision (default: enabled; use --no_bf16 to disable)")
    p.add_argument("--no_bf16", action="store_true", default=False,
                    help="Disable bfloat16 and use float16 instead")

    # Data
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--max_train_samples", type=int, default=None,
                    help="Optional cap on training/cache samples. Default "
                         "uses the full training file.")

    # Student format sampling
    p.add_argument("--format_mode", type=str, default="random_4way",
                    choices=["random_4way", "hetero_only"],
                    help="Format sampling for the activation path: random_4way "
                         "draws each evidence format uniformly from "
                         "text/table/kg/infobox; hetero_only draws one of the "
                         "six heterogeneous format pairs.")
    p.add_argument("--anchor_ratio", type=float, default=0.0,
                    help="Fraction of activation samples kept as all-text anchors")

    # Teacher caching
    p.add_argument("--teacher_cache_dir", type=str, default=None,
                    help="Directory for teacher cache (default: <output_dir>/teacher_cache)")
    p.add_argument("--cache_top_k", type=int, default=256)
    p.add_argument("--cache_batch_size", type=int, default=4)
    p.add_argument("--overwrite_teacher_cache", action="store_true", default=False,
                    help="Overwrite existing teacher cache shards in "
                         "--teacher_cache_dir during cache stage. Default is "
                         "to fail fast to avoid mixing stale logits.")
    p.add_argument("--cache_sample_start", type=int, default=0,
                    help="Cache-stage only: start index after symmetrization. "
                         "Used to split cache construction.")
    p.add_argument("--cache_sample_end", type=int, default=None,
                    help="Cache-stage only: end index after symmetrization. "
                         "Used to split cache construction.")

    # Format-marginalized teacher target
    p.add_argument("--adaptive_backdoor_low", type=float, default=0.01,
                    help="tau_low: mean JS divergence across format "
                         "assignments below this uses the minimum weight "
                         "on the format-marginalized target.")
    p.add_argument("--adaptive_backdoor_high", type=float, default=0.05,
                    help="tau_high: mean JS divergence above this uses the "
                         "maximum weight.")
    p.add_argument("--adaptive_backdoor_min_weight", type=float, default=0.0,
                    help="alpha_min: minimum weight on the format-marginalized target.")
    p.add_argument("--adaptive_backdoor_max_weight", type=float, default=1.0,
                    help="alpha_max: maximum weight on the format-marginalized target.")

    # Loss
    p.add_argument("--top_k", type=int, default=256,
                    help="Student top-K used by the anchoring-path reverse KL")
    p.add_argument("--anchor_coef", type=float, default=0.0,
                    help="lambda_anc: weight of the anchoring-path KL")
    p.add_argument("--gold_ce_coef", type=float, default=0.2,
                    help="lambda_gold: CE on gold response tokens, keeps the "
                         "averaged teacher targets from becoming over-smooth.")
    p.add_argument("--no_symmetric_targets", action="store_true", default=False,
                    help="Disable the deterministic A/B content swap of every "
                         "second training pair (must match between the cache "
                         "and train stages).")
    p.add_argument("--balance_coef", type=float, default=0.1,
                    help="Coefficient for the hidden-state balance loss (0 = disabled).")
    p.add_argument("--format_adv_coef", type=float, default=0.05,
                    help="Coefficient for the GRL format-adversarial loss on "
                         "evidence hidden states (0 = disabled).")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42,
                    help="Global RNG seed. Sets torch/CUDA/numpy/python.random "
                         "seeds. Use -1 to disable seeding.")

    # Training
    p.add_argument("--output_dir", type=str, default="./outputs/fade")
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--no_resume", action="store_true", default=False)

    # LoRA
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    return p.parse_args()


def _resolve_data_paths(args):
    """Fall back to dataset/conflictbank/conflict_train*.jsonl if not set."""
    data_dir = os.path.join(_SCRIPT_DIR, "dataset", "conflictbank")

    def _first(prefix):
        candidates = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith(prefix) and f.endswith(".jsonl")
        ]) if os.path.isdir(data_dir) else []
        return candidates[0] if candidates else os.path.join(data_dir, f"{prefix}.jsonl")

    if args.data_path is None:
        args.data_path = _first("conflict_train")


def _seed_everything(seed: int) -> None:
    """Seed Python / NumPy / PyTorch (CPU+CUDA) RNGs for reproducibility.

    NOTE: We intentionally leave `torch.use_deterministic_algorithms`,
    `torch.backends.cudnn.deterministic`, and `CUBLAS_WORKSPACE_CONFIG`
    OFF because they can break fused attention kernels on some backbones.
    """
    import random as _random
    import numpy as _np
    import torch as _torch
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _load_train_pairs(args):
    from data.conflictbank_processor import load_conflictbank_processed, symmetrize_pairs

    logger.info(f"Loading ConflictBank training data from {args.data_path}")
    conflict_pairs = load_conflictbank_processed(args.data_path)
    if args.max_train_samples and len(conflict_pairs) > args.max_train_samples:
        conflict_pairs = conflict_pairs[:args.max_train_samples]
    logger.info(f"Loaded {len(conflict_pairs)} conflict pairs")

    if args.symmetric_targets:
        logger.info("Applying symmetric targets: deterministically swapping 50% of pairs")
        conflict_pairs = symmetrize_pairs(conflict_pairs)
    return conflict_pairs


def main():
    args = parse_args()
    if args.no_bf16:
        args.bf16 = False
    elif not args.bf16:
        args.bf16 = True
    args.symmetric_targets = not args.no_symmetric_targets

    if args.seed >= 0:
        _seed_everything(args.seed)
        logger.info(f"Global seed set to {args.seed} (deterministic RNG)")

    _resolve_data_paths(args)

    run_cache = args.stage in ["cache", "all"]
    run_train = args.stage in ["train", "all"]

    teacher_cache_dir = args.teacher_cache_dir or os.path.join(args.output_dir, "teacher_cache")

    logger.info(f"Train data: {args.data_path}")

    # --- Stage 0: Cache teacher logits ---
    if run_cache:
        from data.teacher_cache import cache_teacher_logits
        from data.conflictbank_processor import build_teacher_cache_samples

        conflict_pairs = _load_train_pairs(args)

        if args.cache_sample_start or args.cache_sample_end is not None:
            start = max(args.cache_sample_start, 0)
            end = args.cache_sample_end
            if end is None:
                end = len(conflict_pairs)
            end = min(end, len(conflict_pairs))
            if start >= end:
                raise ValueError(
                    f"Invalid cache sample slice [{start}, {end}) for "
                    f"{len(conflict_pairs)} ConflictBank pairs"
                )
            logger.info(
                "Cache sample slice: [%d, %d) of %d pairs",
                start, end, len(conflict_pairs),
            )
            conflict_pairs = conflict_pairs[start:end]

        logger.info("Building all-text and format-intervention teacher prompts...")
        samples = build_teacher_cache_samples(conflict_pairs)

        logger.info(f"Caching teacher logits to {teacher_cache_dir}...")
        cache_teacher_logits(
            model_name=args.model_name,
            samples=samples,
            cache_dir=teacher_cache_dir,
            top_k=args.cache_top_k,
            max_length=args.max_length,
            batch_size=args.cache_batch_size,
            bf16=args.bf16,
            model_cache_dir=args.cache_dir,
            overwrite=args.overwrite_teacher_cache,
            adaptive_backdoor_low=args.adaptive_backdoor_low,
            adaptive_backdoor_high=args.adaptive_backdoor_high,
            adaptive_backdoor_min_weight=args.adaptive_backdoor_min_weight,
            adaptive_backdoor_max_weight=args.adaptive_backdoor_max_weight,
            extra_meta={"symmetric_targets": args.symmetric_targets},
        )
        logger.info("Teacher caching complete.")

    # --- Stage 1: Train ---
    if run_train:
        from trainers.fade_trainer import FaDeTrainer

        # The cached targets are computed on the (optionally swapped) pairs,
        # so both stages must use the same symmetric-target setting.
        meta_path = os.path.join(teacher_cache_dir, "cache_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_symmetric = json.load(f).get("symmetric_targets")
            if cached_symmetric is not None and cached_symmetric != args.symmetric_targets:
                raise ValueError(
                    f"Teacher cache was built with symmetric_targets="
                    f"{cached_symmetric}, but training uses "
                    f"symmetric_targets={args.symmetric_targets}. Use the same "
                    "--no_symmetric_targets setting for both stages."
                )

        trainer = FaDeTrainer(
            model_name=args.model_name,
            teacher_cache_dir=teacher_cache_dir,
            cache_dir=args.cache_dir,
            learning_rate=args.learning_rate,
            anchor_coef=args.anchor_coef,
            top_k=args.top_k,
            max_length=args.max_length,
            bf16=args.bf16,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            balance_coef=args.balance_coef,
            gold_ce_coef=args.gold_ce_coef,
            format_adv_coef=args.format_adv_coef,
        )

        conflict_pairs = _load_train_pairs(args)

        logger.info(f"Starting FaDe training on ConflictBank ({len(conflict_pairs)} pairs)...")
        trainer.train(
            conflict_pairs=conflict_pairs,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_path=args.output_dir,
            save_steps=args.save_steps,
            format_mode=args.format_mode,
            anchor_ratio=args.anchor_ratio,
            resume=not args.no_resume,
        )
        logger.info("Training complete!")


if __name__ == "__main__":
    main()
