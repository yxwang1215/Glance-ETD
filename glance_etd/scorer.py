"""Evidence scorer f_phi (Sec. 3.3, "Evidence scorer" and "Scorer supervision").

Reads the MLLM's own post-merger visual tokens and the question's input embeddings (no extra encoder) and outputs
a relevance logit r_i per visual token (p = softmax r) and an auxiliary text logit (used only by the t-based
ablations in the supplementary). About 1.9M parameters for d = 2560, h = 256.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos(rows: torch.Tensor, cols: torch.Tensor, gh: int, gw: int, dim: int) -> torch.Tensor:
    """2-D sine-cosine code of normalised cell centres, dim/2 channels per axis."""
    y = (rows.float() + 0.5) / gh
    x = (cols.float() + 0.5) / gw
    freq = torch.exp(torch.arange(dim // 4, device=rows.device) * (-math.log(100.0) / max(dim // 4 - 1, 1)))
    parts = []
    for p in (y, x):
        a = p[:, None] * freq[None] * math.pi * 8
        parts += [a.sin(), a.cos()]
    return torch.cat(parts, dim=-1)


class EvidenceScorer(nn.Module):
    """Project visual tokens to h, add a 2-D position code, cross-attend to the projected question tokens, and score
    [h_i, c_i, h_i * mean(q)] with an MLP. Checkpoints store {"state_dict", "hidden", "d"}."""

    def __init__(self, d: int = 2560, h: int = 256, heads: int = 4):
        super().__init__()
        self.h = h
        self.v = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.q = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.pos = nn.Linear(h, h)
        self.cross = nn.MultiheadAttention(h, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(3 * h), nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 2))

    def forward(self, v, q, rows, cols, gh, gw):
        """v [N, d] visual tokens, q [L, d] question embeddings, (rows, cols) grid position of every token in a
        gh x gw grid -> (relevance logits [N], text logits [N])."""
        hv = self.v(v.float()) + self.pos(sincos(rows, cols, gh, gw, self.h))
        hq = self.q(q.float())
        ctx, _ = self.cross(hv[None], hq[None], hq[None])
        ctx = ctx[0]
        pooled = hq.mean(0, keepdim=True).expand_as(hv)
        logits = self.out(torch.cat([hv, ctx, hv * pooled], dim=-1))
        return logits[:, 0], logits[:, 1]

    @classmethod
    def load(cls, path, device="cpu"):
        blob = torch.load(path, map_location="cpu")
        net = cls(d=blob["d"], h=blob["hidden"])
        net.load_state_dict(blob["state_dict"])
        return net.to(device).eval()


def box_coverage(gh: int, gw: int, boxes):
    """Fraction of each grid cell covered by normalised [x0, y0, x1, y1] boxes (max over boxes).

    Returns (coverage [gh*gw], rows, cols) in raster order, the order of Qwen's merged visual tokens."""
    rows = torch.arange(gh).repeat_interleave(gw)
    cols = torch.arange(gw).repeat(gh)
    y0, y1 = rows.float() / gh, (rows + 1).float() / gh
    x0, x1 = cols.float() / gw, (cols + 1).float() / gw
    cover = torch.zeros(gh * gw)
    for bx0, by0, bx1, by1 in boxes:
        ix = (torch.clamp(x1, max=bx1) - torch.clamp(x0, min=bx0)).clamp_min(0)
        iy = (torch.clamp(y1, max=by1) - torch.clamp(y0, min=by0)).clamp_min(0)
        cover = torch.maximum(cover, ix * iy * gh * gw)
    return cover, rows, cols


def listwise_loss(relevance_logits: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
    """-sum_i y_i log p_i with y = coverage / sum(coverage): one image is one list, only the ranking is supervised."""
    target = coverage.to(relevance_logits.device) / coverage.sum().clamp_min(1e-6)
    return -(target * F.log_softmax(relevance_logits.float(), -1)).sum()
