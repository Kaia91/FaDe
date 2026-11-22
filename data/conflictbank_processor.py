"""
ConflictBank dataset loading and format-varied prompt construction.

  - Each sample pairs two contradicting evidence pieces in different formats.
  - 6 heterogeneous format pairs + 1 homogeneous (text-text) control.

Training mode:
  Teacher sees both evidence in text -> produces logits for the factual answer.
  Student sees evidence in heterogeneous formats -> matches teacher distribution.
"""

import json
import random
from typing import List, Tuple


ALL_FORMATS = ["text", "table", "kg", "infobox"]

FORMAT_PAIRS = [
    ("text", "table"), ("text", "kg"), ("text", "infobox"),
    ("table", "kg"), ("table", "infobox"), ("kg", "infobox"),
]

FORMAT_PAIRS_WITH_CONTROL = [("text", "text")] + FORMAT_PAIRS


def load_conflictbank_processed(data_path: str) -> List[dict]:
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

CONFLICT_PROMPT_TEMPLATE = (
    "Based on the two reference sources provided below, "
    "answer the following question **concisely**.\n\n"
    "Source A:\n{source_a}\n\n"
    "Source B:\n{source_b}\n\n"
    "Question: {question}"
)


def build_conflict_prompt(
    pair: dict,
    format_a: str = "text",
    format_b: str = "text",
    randomize_order: bool = True,
) -> dict:
    """Build a prompt for a contradiction pair with specified formats.

    Args:
        pair: A contradiction pair dict with evidence and format variants.
        format_a: Format for evidence A (factual).
        format_b: Format for evidence B (counterfactual).
        randomize_order: If True, randomly swap A/B order to eliminate
                        position bias (following the paper's protocol).
    Returns:
        Dict with prompt, metadata, and ordering info.
    """
    ev_a_key = f"evidence_a_{format_a}" if format_a != "text" else "evidence_a"
    ev_b_key = f"evidence_b_{format_b}" if format_b != "text" else "evidence_b"

    source_a = pair.get(ev_a_key, pair["evidence_a"])
    source_b = pair.get(ev_b_key, pair["evidence_b"])

    swapped = False
    if randomize_order and random.random() < 0.5:
        source_a, source_b = source_b, source_a
        swapped = True

    prompt = CONFLICT_PROMPT_TEMPLATE.format(
        source_a=source_a, source_b=source_b, question=pair["question"],
    )

    return {
        "question": pair["question"],
        "prompt": prompt,
        "claim_a": pair["claim_a"],
        "claim_b": pair["claim_b"],
        "answer_a": pair["answer_a"],
        "answer_b": pair["answer_b"],
        "format_a": format_a,
        "format_b": format_b,
        "swapped": swapped,
        "conflict_type": pair.get("conflict_type", ""),
        "pair_id": pair["id"],
    }


def build_conflict_prompt_all_text(pair: dict) -> dict:
    """Build the all-text (teacher) version of the prompt."""
    source_a = pair["evidence_a"]
    source_b = pair["evidence_b"]

    prompt = CONFLICT_PROMPT_TEMPLATE.format(
        source_a=source_a, source_b=source_b, question=pair["question"],
    )
    return {
        "question": pair["question"],
        "prompt": prompt,
        "prompt_all_text": prompt,
        "answer_a": pair["answer_a"],
        "answer_b": pair["answer_b"],
    }


def build_backdoor_format_pairs() -> List[Tuple[str, str]]:
    """Return the format assignments used to marginalize the teacher target.

    The format-marginalized cache averages teacher predictions over this
    finite support:

        P(Y | do(Content=c)) ~= sum_f P_T(Y | C=c, F=f) pi(f)

    It is the 7-way ConflictBank evaluation support (text-text control plus
    the six heterogeneous pairs).
    """
    return list(FORMAT_PAIRS_WITH_CONTROL)


def build_backdoor_teacher_variants_conflict(pair: dict) -> List[dict]:
    """Build same-content teacher prompts under multiple format interventions.

    Random order swaps are deliberately disabled; only the format assignment
    changes.
    """
    variants = []
    for fmt_a, fmt_b in build_backdoor_format_pairs():
        prompt_data = build_conflict_prompt(
            pair, format_a=fmt_a, format_b=fmt_b, randomize_order=False,
        )
        variants.append({
            "prompt": prompt_data["prompt"],
            "format_pair": [fmt_a, fmt_b],
        })
    return variants


# ---------------------------------------------------------------------------
# Training data construction
# ---------------------------------------------------------------------------

def build_teacher_cache_samples(pairs: List[dict]) -> List[dict]:
    """Build the teacher-cache inputs for FaDe training on ConflictBank.

    Teacher: both evidence in text format, plus the same content under each
    format assignment of the marginalization support.
    Answer: the factual answer (answer_a) for distillation target.
    """
    samples = []
    for pair in pairs:
        variants = build_backdoor_teacher_variants_conflict(pair)
        samples.append({
            "question": pair["question"],
            "answer": pair["answer_a"],
            "teacher_prompt": build_conflict_prompt_all_text(pair)["prompt"],
            "teacher_prompts_backdoor": [v["prompt"] for v in variants],
            "teacher_format_pairs_backdoor": [v["format_pair"] for v in variants],
            "pair_id": pair["id"],
        })
    return samples


def _swap_pair(pair: dict) -> dict:
    """Swap evidence A/B and their answers to symmetrize training targets."""
    swapped = dict(pair)
    swapped["evidence_a"] = pair["evidence_b"]
    swapped["evidence_b"] = pair["evidence_a"]
    swapped["claim_a"] = pair["claim_b"]
    swapped["claim_b"] = pair["claim_a"]
    swapped["answer_a"] = pair.get("answer_b", "")
    swapped["answer_b"] = pair.get("answer_a", "")
    for fmt in ("table", "kg", "infobox"):
        key_a = f"evidence_a_{fmt}"
        key_b = f"evidence_b_{fmt}"
        if key_a in pair or key_b in pair:
            swapped[key_a] = pair.get(key_b, "")
            swapped[key_b] = pair.get(key_a, "")
    return swapped


def symmetrize_pairs(pairs: List[dict]) -> List[dict]:
    """Deterministically swap 50% of pairs to cancel content preference bias.

    Even-indexed pairs keep original order (answer_a target);
    odd-indexed pairs swap A<->B (answer_b becomes new answer_a).
    This ensures cached teacher logits match the training data.
    """
    result = []
    for i, pair in enumerate(pairs):
        if i % 2 == 1:
            result.append(_swap_pair(pair))
        else:
            result.append(pair)
    return result


def _sample_format_pair(mode: str) -> Tuple[str, str]:
    """Sample a format pair for the two evidence pieces."""
    if mode == "random_4way":
        return random.choice(ALL_FORMATS), random.choice(ALL_FORMATS)
    if mode == "hetero_only":
        return random.choice(FORMAT_PAIRS)
    raise ValueError(f"Unknown format mode: {mode}")
