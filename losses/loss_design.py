"""
Loss functions for FaDe.

Teacher targets are pre-computed top-K log-probs loaded from the offline
cache. Asymmetric KL design:
  - Activation path (heterogeneous -> all-text): TopKForwardKL
    - Compatible with cached teacher top-K (teacher's high-prob tokens drive loss)
    - Mode-covering ensures student covers all critical teacher tokens
  - Anchoring path  (all-text -> all-text): TopKReverseKL
    - Student and teacher distributions are close -> high top-K overlap
    - Mode-seeking gently maintains existing capability without over-fitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100


# ===========================================================================
# Top-K KL losses on cached teacher targets
# ===========================================================================

class TopKForwardKLLoss(nn.Module):
    """Forward KL on teacher's top-K tokens.

    The teacher provides pre-computed top-K indices and renormalized
    log-probs. We gather the corresponding student log-probs and compute KL
    on this sub-vocabulary.
    """

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_topk_indices: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        mask = (labels != IGNORE_INDEX).float()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        s = student_logits[:, :-1, :].contiguous()
        m = mask[:, 1:].contiguous()

        t_topk_idx = teacher_topk_indices[:, :-1, :].contiguous().long()
        t_topk_lp = teacher_topk_logprobs[:, :-1, :].contiguous().detach()
        t_topk_probs = t_topk_lp.exp()

        s_log = F.log_softmax(s, dim=-1)
        s_topk_log = s_log.gather(-1, t_topk_idx)

        kl = F.kl_div(s_topk_log, t_topk_probs, reduction="none").sum(dim=-1)
        return (kl * m).sum() / m.sum().clamp(min=1)


class TopKReverseKLLoss(nn.Module):
    """Reverse KL on student's top-K tokens.

    Mode-seeking behavior: focuses on tokens the student assigns high
    probability to. Penalizes the student for confidently predicting tokens
    that the teacher considers unlikely.
    """

    def __init__(self, top_k: int = 256):
        super().__init__()
        self.top_k = top_k

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_topk_indices: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        mask = (labels != IGNORE_INDEX).float()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        s = student_logits[:, :-1, :].contiguous()
        m = mask[:, 1:].contiguous()

        s_log_full = F.log_softmax(s, dim=-1)
        _, s_topk_idx = s_log_full.topk(self.top_k, dim=-1)

        s_topk_log = s_log_full.gather(-1, s_topk_idx)
        s_topk_log = s_topk_log - torch.logsumexp(s_topk_log, dim=-1, keepdim=True)

        t_cached_idx = teacher_topk_indices[:, :-1, :].contiguous().long()
        t_cached_lp = teacher_topk_logprobs[:, :-1, :].contiguous().detach()
        t_topk_log = _gather_from_cached(s_topk_idx, t_cached_idx, t_cached_lp)

        t_topk_log = t_topk_log - torch.logsumexp(t_topk_log, dim=-1, keepdim=True)

        s_topk_probs = s_topk_log.exp()
        kl = (s_topk_probs * (s_topk_log - t_topk_log)).sum(dim=-1)
        return (kl * m).sum() / m.sum().clamp(min=1)


def _gather_from_cached(
    query_idx: torch.Tensor,
    cached_idx: torch.Tensor,
    cached_logprobs: torch.Tensor,
    default_logprob: float = -20.0,
) -> torch.Tensor:
    """Look up query_idx in cached_idx; return cached logprobs or default.

    For reverse KL with cached teacher, the student's top-K tokens may not
    overlap with the teacher's cached top-K. Tokens not found in the cache
    receive a very low log-probability (effectively zero probability), which
    correctly penalizes the student for being confident about tokens the
    teacher assigns negligible mass to.
    """
    B, L, Kq = query_idx.shape
    _, _, Kc = cached_idx.shape
    device = query_idx.device

    result = torch.full((B, L, Kq), default_logprob, device=device, dtype=cached_logprobs.dtype)

    q_exp = query_idx.unsqueeze(-1).expand(B, L, Kq, Kc)
    c_exp = cached_idx.unsqueeze(-2).expand(B, L, Kq, Kc)
    match = (q_exp == c_exp)
    found_mask = match.any(dim=-1)

    match_idx = match.float().argmax(dim=-1)
    matched_lp = cached_logprobs.gather(-1, match_idx)
    result[found_mask] = matched_lp[found_mask]

    return result


# ===========================================================================
# Hidden-state balance loss used as an attention balance proxy
# ===========================================================================

class HiddenStateBalanceLoss(nn.Module):
    """Penalizes imbalanced processing of two evidence segments.

    Uses intermediate hidden-state norms as a proxy for attention allocation.
    If the model amplifies one segment's representations far more than the
    other, this loss pushes toward equal processing intensity.  Compatible
    with flash_attention_2 (only needs output_hidden_states=True).
    """

    def __init__(self, layer_indices=None):
        super().__init__()
        self.layer_indices = layer_indices

    def forward(
        self,
        hidden_states: tuple,
        seg_a_mask: torch.Tensor,
        seg_b_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: tuple of (B, L, D) from model(output_hidden_states=True)
            seg_a_mask: (B, L) bool - tokens belonging to evidence A
            seg_b_mask: (B, L) bool - tokens belonging to evidence B
        """
        n_layers = len(hidden_states)
        indices = self.layer_indices
        if indices is None:
            indices = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

        loss = torch.tensor(0.0, device=hidden_states[0].device, requires_grad=True)
        count = 0

        a_float = seg_a_mask.float()
        b_float = seg_b_mask.float()
        a_count = a_float.sum(dim=1, keepdim=True).clamp(min=1)
        b_count = b_float.sum(dim=1, keepdim=True).clamp(min=1)

        valid = (a_count.squeeze() > 0) & (b_count.squeeze() > 0)
        if not valid.any():
            return loss

        for li in indices:
            if li >= n_layers:
                continue
            h = hidden_states[li]
            norms = h.norm(dim=-1)

            mean_a = (norms * a_float).sum(dim=1) / a_count.squeeze()
            mean_b = (norms * b_float).sum(dim=1) / b_count.squeeze()

            total = mean_a + mean_b + 1e-8
            imbalance = torch.abs(mean_a - mean_b) / total
            loss = loss + imbalance[valid].mean()
            count += 1

        return loss / max(count, 1)


# ===========================================================================
# Gold response CE and format-adversarial deconfounding
# ===========================================================================

class GoldResponseCELoss(nn.Module):
    """CE on the gold response tokens to prevent averaged targets diluting."""

    def forward(
        self,
        student_logits: torch.Tensor,
        target_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        if target_ids is None:
            return student_logits.new_zeros(())

        s = student_logits[:, :-1, :].contiguous()
        targets = target_ids[:, 1:].contiguous()

        if labels is not None:
            valid = labels[:, 1:].ne(IGNORE_INDEX)
            targets = targets.masked_fill(~valid, IGNORE_INDEX)

        if targets.ne(IGNORE_INDEX).sum() == 0:
            return student_logits.new_zeros(())

        return F.cross_entropy(
            s.float().view(-1, s.size(-1)),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
        )


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def _grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradientReverse.apply(x, lambd)


class FormatAdversarialLoss(nn.Module):
    """Make evidence representations predictive for task but not format.

    A lightweight classifier predicts the rendered format of Source A/B from
    pooled final-layer hidden states. A gradient reversal layer lets the
    classifier learn the format signal while pushing the student
    representation to remove it.
    """

    def __init__(
        self,
        hidden_size: int,
        num_formats: int = 4,
        layer_idx: int = -1,
        grl_lambda: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.grl_lambda = grl_lambda
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_formats),
        )

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        hidden_states: tuple,
        seg_a_mask: torch.Tensor,
        seg_b_mask: torch.Tensor,
        format_labels: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states is None or format_labels is None:
            device = seg_a_mask.device if seg_a_mask is not None else "cpu"
            return torch.zeros((), device=device)

        n_layers = len(hidden_states)
        layer_idx = self.layer_idx
        if layer_idx < 0:
            layer_idx = n_layers + layer_idx
        layer_idx = min(max(layer_idx, 0), n_layers - 1)

        h = hidden_states[layer_idx]
        pooled_a = self._masked_mean(h, seg_a_mask)
        pooled_b = self._masked_mean(h, seg_b_mask)

        pooled = torch.cat([pooled_a, pooled_b], dim=0)
        labels = torch.cat([format_labels[:, 0], format_labels[:, 1]], dim=0)
        valid_masks = torch.cat(
            [seg_a_mask.any(dim=1), seg_b_mask.any(dim=1)], dim=0,
        )
        valid = valid_masks & labels.ne(IGNORE_INDEX)

        if not valid.any():
            return h.new_zeros(())

        cls_dtype = next(self.classifier.parameters()).dtype
        reversed_repr = _grad_reverse(pooled[valid].to(cls_dtype), self.grl_lambda)
        logits = self.classifier(reversed_repr)
        return F.cross_entropy(logits.float(), labels[valid].long())


# ===========================================================================
# Combined loss
# ===========================================================================

class FaDeLoss(nn.Module):
    """Combined loss for FaDe.

    L = L_act + lambda_anc * L_anc + lambda_gold * L_gold
        + lambda_bal * L_bal + lambda_adv * L_adv

      - L_act (activation): student sees heterogeneous formats -> align to the
        format-marginalized teacher target
      - L_anc (anchoring):  student sees all-text -> maintain baseline ability
      - L_gold: CE on the gold response tokens, averaged over both paths
      - L_bal (balance):    penalize imbalanced processing of evidence segments
      - L_adv: format classifier behind a gradient reversal layer
    """

    def __init__(
        self,
        anchor_coef: float = 0.0,
        top_k: int = 256,
        balance_coef: float = 0.1,
        gold_ce_coef: float = 0.2,
        format_adv_coef: float = 0.05,
        format_adv_hidden_size: int = None,
    ):
        super().__init__()
        self.anchor_coef = anchor_coef
        self.balance_coef = balance_coef
        self.gold_ce_coef = gold_ce_coef
        self.format_adv_coef = format_adv_coef
        self.act_loss_fn = TopKForwardKLLoss()
        self.anc_loss_fn = TopKReverseKLLoss(top_k=top_k)
        self.balance_loss_fn = HiddenStateBalanceLoss() if balance_coef > 0 else None
        self.gold_ce_loss_fn = GoldResponseCELoss() if gold_ce_coef > 0 else None
        if format_adv_coef > 0 and format_adv_hidden_size is None:
            raise ValueError("format_adv_hidden_size is required when format_adv_coef > 0")
        self.format_adv_loss_fn = (
            FormatAdversarialLoss(hidden_size=format_adv_hidden_size)
            if format_adv_coef > 0 else None
        )

    def forward(
        self,
        student_act_logits: torch.Tensor,
        student_anc_logits: torch.Tensor,
        labels_act: torch.Tensor,
        labels_anc: torch.Tensor,
        teacher_topk_indices_act: torch.Tensor,
        teacher_topk_logprobs_act: torch.Tensor,
        teacher_topk_indices_anc: torch.Tensor,
        teacher_topk_logprobs_anc: torch.Tensor,
        hidden_states_act: tuple = None,
        seg_a_mask: torch.Tensor = None,
        seg_b_mask: torch.Tensor = None,
        target_ids_act: torch.Tensor = None,
        target_ids_anc: torch.Tensor = None,
        format_labels: torch.Tensor = None,
    ) -> dict:
        loss_act = self.act_loss_fn(
            student_act_logits, labels_act,
            teacher_topk_indices_act, teacher_topk_logprobs_act,
        )
        loss_anc = self.anc_loss_fn(
            student_anc_logits, labels_anc,
            teacher_topk_indices_anc, teacher_topk_logprobs_anc,
        )

        total = loss_act + self.anchor_coef * loss_anc

        loss_gold_ce = torch.tensor(0.0, device=total.device)
        if self.gold_ce_loss_fn is not None:
            ce_parts = []
            if target_ids_act is not None:
                ce_parts.append(
                    self.gold_ce_loss_fn(
                        student_act_logits, target_ids_act, labels_act,
                    )
                )
            if target_ids_anc is not None:
                ce_parts.append(
                    self.gold_ce_loss_fn(
                        student_anc_logits, target_ids_anc, labels_anc,
                    )
                )
            if ce_parts:
                loss_gold_ce = sum(ce_parts) / len(ce_parts)
                total = total + self.gold_ce_coef * loss_gold_ce

        loss_bal = torch.tensor(0.0, device=total.device)
        if (self.balance_loss_fn is not None
                and hidden_states_act is not None
                and seg_a_mask is not None):
            loss_bal = self.balance_loss_fn(hidden_states_act, seg_a_mask, seg_b_mask)
            total = total + self.balance_coef * loss_bal

        loss_fmt_adv = torch.tensor(0.0, device=total.device)
        if (self.format_adv_loss_fn is not None
                and hidden_states_act is not None
                and seg_a_mask is not None
                and seg_b_mask is not None
                and format_labels is not None):
            loss_fmt_adv = self.format_adv_loss_fn(
                hidden_states_act, seg_a_mask, seg_b_mask, format_labels,
            )
            total = total + self.format_adv_coef * loss_fmt_adv

        return {
            "loss": total,
            "loss_act": loss_act,
            "loss_anc": loss_anc,
            "loss_gold_ce": loss_gold_ce,
            "loss_bal": loss_bal,
            "loss_fmt_adv": loss_fmt_adv,
        }
