"""Stage 2: Evidence-Tempered Diversity (ETD), Sec. 3.3, Eq. (1)-(2).

DivPrune greedily adds the token with the largest minimum cosine distance to the kept set,

    i* = argmax_{i not in S} d_i,        d_i = min_{j in S} (1 - cos(x_i, x_j)).

ETD multiplies the distance by the token's evidence relative to a uniform map, tempered by gamma:

    i* = argmax_{i not in S} d_i * (N p_i) ** gamma,     p = softmax(r),  gamma = 0.25.

With a uniform p every weight is 1 and ETD is exactly DivPrune.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

GAMMA = 0.25
LOG_FLOOR = -80.0  # exp(-80) > 0 in float32; the greedy requires strictly positive multipliers


def strict_budget(num_tokens: int, ratio: float = 0.05) -> int:
    """Strict rule: K = ceil(r N), clamped to [1, N]."""
    return int(min(num_tokens, max(1, math.ceil(ratio * num_tokens - 1e-9))))


def floor_budget(num_tokens: int, ratio: float = 0.05, floor: int = 32) -> int:
    """Floor-32 rule of the LT-OPD release: K = max(32, ceil(r N)), clamped to N."""
    return int(min(num_tokens, max(floor, math.ceil(ratio * num_tokens - 1e-9))))


def power_budget(num_tokens: int, scale: float, exponent: float = 0.45) -> int:
    """Per-image allocation K proportional to N^0.45 (Sec. 3.4 / supplementary). ``scale`` is chosen label-free so
    that sum K / sum N matches the target budget on a benchmark."""
    ratio = min(1.0, scale * num_tokens ** (exponent - 1.0))
    return strict_budget(num_tokens, ratio)


def evidence_weights(relevance_logits: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    """Per-token multipliers (N p_i)^gamma, computed in log space and shifted so that the largest weight is 1.

    argmax(d * w) is invariant to a common positive factor, so the shift keeps the ranking exact for any gamma.
    """
    r = relevance_logits.detach().float().reshape(-1)
    n = r.numel()
    if gamma == 0.0:
        return torch.ones_like(r)
    logw = gamma * (torch.log_softmax(r, 0) + math.log(n))
    return torch.exp((logw - logw.max()).clamp_min(LOG_FLOOR))


def max_min_diversity(
    normalized_visual: torch.Tensor,
    num_keep: int,
    *,
    importance: Optional[torch.Tensor] = None,
    seed: str = "centroid",
) -> torch.Tensor:
    """Greedy max-min cosine diversity (DivPrune) with optional positive per-token multipliers.

    normalized_visual: [N, D] L2-normalised tokens. Returns the kept indices, sorted (original token order).
    seed="centroid": the first token is the one least similar to the mean token, as in DivPrune.
    Ties in argmax resolve to the lowest index. Memory is O(N): distances are updated one row at a time.
    """
    visual = normalized_visual.to(torch.float32)
    n = int(visual.shape[0])
    if not 1 <= num_keep <= n:
        raise ValueError(f"num_keep must be in [1, {n}], got {num_keep}")
    selected = torch.empty(num_keep, device=visual.device, dtype=torch.long)
    if seed == "centroid":
        selected[0] = torch.argmin(torch.mv(visual, visual.mean(dim=0)))
    elif seed == "nearest_neighbor":
        sim = visual @ visual.T
        sim.fill_diagonal_(-torch.inf)
        selected[0] = torch.argmin(sim.max(dim=1).values) if n > 1 else 0
    else:
        raise ValueError(f"unknown seed {seed!r}")
    min_distance = 1.0 - torch.mv(visual, visual[selected[0]])
    min_distance[selected[0]] = -torch.inf
    if importance is not None:
        importance = importance.detach().to(device=visual.device, dtype=torch.float32).reshape(-1)
        if importance.numel() != n or not bool((importance > 0).all()):
            raise ValueError("importance must hold one positive multiplier per token")
    for step in range(1, num_keep):
        if importance is None:
            pivot = torch.argmax(min_distance)
        else:
            # Unselected tokens have distance >= 0 (clamped for fp noise); selected tokens stay at -inf.
            weighted = torch.where(torch.isinf(min_distance), min_distance, min_distance.clamp(min=0.0) * importance)
            pivot = torch.argmax(weighted)
        selected[step] = pivot
        min_distance = torch.minimum(min_distance, 1.0 - torch.mv(visual, visual[pivot]))
        min_distance[selected[: step + 1]] = -torch.inf
    return torch.sort(selected).values


def divprune_select(tokens: torch.Tensor, num_keep: int) -> torch.Tensor:
    """DivPrune on raw (unnormalised) visual tokens [N, D]."""
    return max_min_diversity(torch.nn.functional.normalize(tokens.float(), dim=-1), num_keep)


def etd_select(tokens: torch.Tensor, relevance_logits: torch.Tensor, num_keep: int, gamma: float = GAMMA) -> torch.Tensor:
    """ETD on raw visual tokens [N, D] with scorer logits r [N]. Returns sorted kept indices."""
    n = int(tokens.shape[0])
    if num_keep >= n:
        return torch.arange(n, device=tokens.device)
    weights = evidence_weights(relevance_logits.to(tokens.device), gamma)
    return max_min_diversity(torch.nn.functional.normalize(tokens.float(), dim=-1), num_keep, importance=weights)
