<h1 align="center">FaDe · Format-Averaged Debiasing</h1>

<p align="center">
  <strong>Format as Nuisance: Intervention-Inspired Cross-Format Distillation for Knowledge Conflicts</strong><br>
  Yue Yu, Wang Luo, Shenghao Ye, Zepeng Bao, Jiacheng Liu, Yongqi Li, Wenxin Tang, and Tieyun Qian<br>
  <strong>EMNLP 2026</strong>
</p>

<p align="center">
  <a href="https://aclanthology.org/2026.emnlp-main.2028/">Paper</a> ·
  <a href="https://aclanthology.org/2026.emnlp-main.2028v2.pdf">PDF</a> ·
  <a href="#results">Results</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="docs/format-swap.svg" width="760" alt="In the paper’s example, the model answers London when London is in a table, then Paris when Paris is in the table. Both sources keep their original claims.">
</p>

Two biographies disagree about a scientist’s birthplace. One says Paris, the other London. If you saw only the first answer, you might think the model had found a reason to trust the London account. In the second prompt, Paris moves into the table and the model switches its answer to Paris. [Figure 1 in the paper](https://aclanthology.org/2026.emnlp-main.2028v2.pdf#page=1)

**FaDe trains a language model to give conflicting sources a more consistent reading across formats.** During training, a frozen teacher reads the same claims several ways. At inference, the student reads the evidence once, in the formats it receives.

## Results

Across four model backbones and three conflict datasets, **FaDe reduced format bias by 34.1% on average** relative to the base models. The evaluation covers ConflictBank and controlled conflicts built from MuSiQue and HotpotQA. [Main results, Table 1](https://aclanthology.org/2026.emnlp-main.2028v2.pdf)

The birthplace example has a measurable counterpart. On 1,000 ConflictBank pairs, we tested whether Qwen3-8B would choose one source in one rendering and the opposite source in another. After training, fewer pairs produced such a flip, and more responses acknowledged both accounts. [Tables 1 and 10](https://aclanthology.org/2026.emnlp-main.2028v2.pdf)

| Qwen3-8B on ConflictBank | Base | FaDe |
| :--- | ---: | ---: |
| Pairs with a source flip across renderings ↓ | 60.5% | **41.9%** |
| Responses acknowledging both sources ↑ | 30.9% | **42.0%** |
| Directional format bias ↓ | 0.276 | **0.210** |

Directional bias measures how far a format’s preference rate among single-source answers departs from an even 50–50 split. [Evaluation metrics, §4.1](https://aclanthology.org/2026.emnlp-main.2028v2.pdf)

## How FaDe works

1. **Change how each source is written.** Keep both claims and their order fixed. The frozen teacher reads seven versions of the prompt, one all-text control and six pairings drawn from prose, tables, knowledge-graph triples, and infoboxes. [Prompt construction](data/conflictbank_processor.py)
2. **Build a target for each conflict.** Average the teacher’s next-token distributions across the seven versions. When the teacher gives similar predictions across formats, keep the target close to its all-text prediction. As disagreement grows, give the average more weight. Cache the top 256 tokens per response position. [Teacher cache](data/teacher_cache.py)
3. **Train the student on sampled formats.** A LoRA student sees a fresh choice of formats as it trains and learns to match the cached target. The teacher’s extra readings happen offline. The merged student can answer a single prompt at inference. [Training loop](trainers/fade_trainer.py) · [Losses](losses/loss_design.py)

## Quick start

Use a chat model and a CUDA GPU with room for the model weights, LoRA adapters, and activations. The tested 8B setup used Python 3.10, PyTorch 2.6.0 (CUDA 12.4), Transformers 5.6.2, and PEFT 0.19.1. FlashAttention is optional. PyTorch SDPA is the fallback.

```bash
git clone https://github.com/Kaia91/FaDe.git
cd FaDe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Prepare the training data

Supply a JSONL file with one contradiction pair per line. Each source has a claim, its answer, and the evidence that supports it. Include each available rendering for both sources. If you omit a format for both, the loader uses their prose for that format.

<details>
<summary><strong>Example training pair</strong></summary>

This fictional record follows the birthplace example above. Save it as `pair.json` and run `python3 -m json.tool --compact pair.json > example.jsonl` to write one JSONL record.

```json
{
  "id": "birthplace_001",
  "question": "Where was the scientist Mira Vale born?",
  "claim_a": "Mira Vale was born in Paris.",
  "claim_b": "Mira Vale was born in London.",
  "answer_a": "Paris",
  "answer_b": "London",
  "evidence_a": "The biography gives Paris as Mira Vale's birthplace.",
  "evidence_b": "The biography gives London as Mira Vale's birthplace.",
  "evidence_a_table": "| Scientist | Birthplace |\n| --- | --- |\n| Mira Vale | Paris |",
  "evidence_b_table": "| Scientist | Birthplace |\n| --- | --- |\n| Mira Vale | London |",
  "evidence_a_kg": "(Mira Vale, place of birth, Paris)",
  "evidence_b_kg": "(Mira Vale, place of birth, London)",
  "evidence_a_infobox": "Name: Mira Vale\nBorn: Paris",
  "evidence_b_infobox": "Name: Mira Vale\nBorn: London",
  "conflict_type": "birthplace"
}
```

</details>

### Cache the teacher and train the student

Set `TRAIN_DATA` to your prepared file and choose a Hugging Face model ID or local model directory. Run both commands from the repository root.

```bash
export MODEL_NAME_OR_PATH=Qwen/Qwen3-8B
export TRAIN_DATA=./train.jsonl
export OUTPUT_DIR=outputs/fade

bash scripts/run/run_fade_cache.sh
bash scripts/run/run_fade_train.sh
```

The first command writes targets to `$OUTPUT_DIR/teacher_cache/`. The second saves the merged model and tokenizer in `$OUTPUT_DIR/` and the adapter in `$OUTPUT_DIR/lora_adapter/`.

<details>
<summary><strong>Run both stages in one process or try a subset</strong></summary>

```bash
python3 train.py --stage all \
  --model_name "$MODEL_NAME_OR_PATH" \
  --data_path "$TRAIN_DATA" \
  --output_dir "$OUTPUT_DIR"
```

For a smaller run with the shell scripts, set `MAX_TRAIN_SAMPLES` before running both stages. Keep the same ordered pairs, sample limit, and symmetric-target setting when reusing cached targets. Choose a fresh output directory for a new run, or set `OVERWRITE_TEACHER_CACHE=1` on the cache script to deliberately rebuild an existing cache.

To test the pipeline on the single example above, use a batch size and accumulation interval of one so the optimizer takes a step.

```bash
python3 train.py --stage all --model_name "$MODEL_NAME_OR_PATH" \
  --data_path example.jsonl --output_dir outputs/fade-example \
  --batch_size 1 --gradient_accumulation_steps 1 --num_epochs 1
```

</details>

<details>
<summary><strong>Training defaults and command-line flags</strong></summary>

These are the defaults in the released training entry point. [All training options](train.py)

| Setting | Default | Flag |
| :--- | :--- | :--- |
| LoRA rank / alpha / dropout | 64 / 128 / 0.05 | `--lora_r`, `--lora_alpha`, `--lora_dropout` |
| Learning rate / epochs | 3e-6 / 2 | `--learning_rate`, `--num_epochs` |
| Batch size × accumulation | 2 × 16 | `--batch_size`, `--gradient_accumulation_steps` |
| Cached teacher top-K | 256 | `--cache_top_k` |
| Disagreement thresholds | 0.01 / 0.05 | `--adaptive_backdoor_low`, `--adaptive_backdoor_high` |
| Minimum / maximum weight on the average | 0 / 1 | `--adaptive_backdoor_min_weight`, `--adaptive_backdoor_max_weight` |
| Anchoring KL weight / all-text anchor fraction | 0 / 0 | `--anchor_coef`, `--anchor_ratio` |
| Gold-response cross-entropy weight | 0.2 | `--gold_ce_coef` |
| Symmetric targets | On | `--no_symmetric_targets` to disable |
| Hidden-state balance / format-adversarial weight | 0.1 / 0.05 | `--balance_coef`, `--format_adv_coef` |
| Student format sampling | Uniform per source over four formats | `--format_mode` |
| Random seed | 42 | `--seed` |

</details>

## Citation

```bibtex
@inproceedings{yu-etal-2026-format,
  title     = {Format as Nuisance: Intervention-Inspired Cross-Format Distillation for Knowledge Conflicts},
  author    = {Yu, Yue and Luo, Wang and Ye, Shenghao and Bao, Zepeng and
               Liu, Jiacheng and Li, Yongqi and Tang, Wenxin and Qian, Tieyun},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  pages     = {46190--46211},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/2026.emnlp-main.2028},
  url       = {https://aclanthology.org/2026.emnlp-main.2028/}
}
```

The code is available under the [MIT License](LICENSE).
