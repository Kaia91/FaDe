# FaDe

Code for the EMNLP 2026 paper *Format as Nuisance: Intervention-Inspired
Cross-Format Distillation for Knowledge Conflicts*.

FaDe trains a LoRA student so that its answer to two conflicting pieces of
evidence does not depend on the formats (text, table, knowledge-graph triples,
infobox) the evidence is written in. A frozen copy of the base model is queried
on the same content under several format assignments; the averaged response
distribution, mixed with the original all-text distribution according to how
much the formats disagree, is the distillation target for the student when it
reads the evidence in sampled formats.

This repository contains the core FaDe implementation: teacher-cache
construction and training.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `train.py` | Entry point for the two stages: teacher cache and training. |
| `data/` | Data loading, format-varied prompt construction, teacher cache. |
| `losses/` | FaDe training losses. |
| `trainers/` | Training loop and checkpointing. |
| `scripts/run/` | Shell wrappers for the two stages. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.6.0 (CUDA 12.4), transformers 5.6.2 and
peft 0.19.1. `flash-attn` is optional; without it the code falls back to
PyTorch SDPA attention. Training an 8B model needs one GPU that holds the bf16
model, the LoRA adapters and the activations; the teacher is only loaded during
the cache stage.

## Input Data

Training files are JSONL, one contradiction pair per line:

```json
{
  "id": "entry_12_temporal",
  "question": "...",
  "claim_a": "...",
  "claim_b": "...",
  "evidence_a": "...",
  "evidence_b": "...",
  "answer_a": "...",
  "answer_b": "...",
  "conflict_type": "temporal",
  "evidence_a_table": "...",
  "evidence_a_kg": "...",
  "evidence_a_infobox": "...",
  "evidence_b_table": "...",
  "evidence_b_kg": "...",
  "evidence_b_infobox": "..."
}
```

`evidence_a` supports `answer_a` and `evidence_b` supports `answer_b`. The
`*_table`, `*_kg` and `*_infobox` fields hold the same evidence rendered in
that format; if one is missing, the plain-text evidence is used instead.

## Running FaDe

```bash
export MODEL_NAME_OR_PATH=<huggingface-model-id-or-local-path>
export TRAIN_DATA=dataset/conflictbank/conflict_train.jsonl
export OUTPUT_DIR=outputs/fade

# Stage 0: cache the frozen teacher's targets
bash scripts/run/run_fade_cache.sh

# Stage 1: train the LoRA student (saves the adapter and a merged model)
bash scripts/run/run_fade_train.sh
```

`python3 train.py --stage all --model_name ... --data_path ...` runs both
stages in one process. For a quick check on a subset, set the
same `MAX_TRAIN_SAMPLES` for the cache and training stages; the training stage
checks that the cache was built from the same pairs and with the same
symmetric-target setting. An existing cache directory is never overwritten
unless `OVERWRITE_TEACHER_CACHE=1` is set.

## Default Configuration

The defaults of `train.py` are the settings of our FaDe runs; the same
settings were used for every backbone.

| Setting | Default | Flag |
| --- | --- | --- |
| LoRA rank / alpha / dropout | 64 / 128 / 0.05 | `--lora_r`, `--lora_alpha`, `--lora_dropout` |
| Learning rate | 3e-6 | `--learning_rate` |
| Batch size x gradient accumulation | 2 x 16 | `--batch_size`, `--gradient_accumulation_steps` |
| Epochs | 2 | `--num_epochs` |
| Cached teacher top-K | 256 | `--cache_top_k` |
| Format assignments for the marginalized target | text-text + 6 heterogeneous pairs | fixed |
| tau_low / tau_high | 0.01 / 0.05 | `--adaptive_backdoor_low`, `--adaptive_backdoor_high` |
| alpha_min / alpha_max | 0 / 1 | `--adaptive_backdoor_min_weight`, `--adaptive_backdoor_max_weight` |
| Anchoring KL weight lambda_anc | 0 | `--anchor_coef` |
| Fraction of all-text activation samples | 0 | `--anchor_ratio` |
| Gold-response CE weight lambda_gold | 0.2 | `--gold_ce_coef` |
| Symmetric targets | on | `--no_symmetric_targets` |
| Hidden-state balance weight | 0.1 | `--balance_coef` |
| Format-adversarial weight | 0.05 | `--format_adv_coef` |
| Student format sampling | each source uniform over the 4 formats | `--format_mode` |
| Seed | 42 | `--seed` |

## Citation

```bibtex
@inproceedings{fade2026,
  title     = {Format as Nuisance: Intervention-Inspired Cross-Format Distillation for Knowledge Conflicts},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

The full entry will be added once the proceedings are online.

## License

MIT, see `LICENSE`.
