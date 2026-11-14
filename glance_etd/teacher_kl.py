"""Teacher-KL diagnostic (Sec. 3.4, Eq. (3)).

For a full-token greedy answer A (at most 32 tokens), the cost of a kept subset S is

    L(S) = sum_a KL( pi(. | X, q, A_<a) || pi(. | X_S, q, A_<a) ),

computed with teacher forcing along A on the untrained model: one forward pass with all visual tokens and one with
the subset S, both on the same prompt + answer tokens (answer positions continue the prompt's M-RoPE text positions).
"""
import torch
import torch.nn.functional as F


def teacher_kl(full_logits: torch.Tensor, subset_logits: torch.Tensor) -> float:
    """Next-token logits [|A|, V] at the answer positions of the two passes -> summed KL(full || subset)."""
    lp = F.log_softmax(full_logits.float(), -1)
    lq = F.log_softmax(subset_logits.float(), -1)
    return float((lp.exp() * (lp - lq)).sum())
