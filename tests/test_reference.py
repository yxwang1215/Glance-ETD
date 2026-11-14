"""CPU tests of the reference implementation: python -m pytest tests (or python tests/test_reference.py)."""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glance_etd.etd import (etd_select, evidence_weights, floor_budget, max_min_diversity, power_budget,  # noqa: E402
                            strict_budget)
from glance_etd.glance import diversity_picker, glance_encode_one, scorer_picker  # noqa: E402
from glance_etd.scorer import EvidenceScorer, box_coverage, listwise_loss  # noqa: E402
from glance_etd.teacher_kl import teacher_kl  # noqa: E402


def naive_divprune(x, k, w=None):
    """O(N^2) reference: centroid seed, then argmax_i min_j (1 - cos) * w_i."""
    x = F.normalize(x.float(), dim=-1)
    w = torch.ones(x.shape[0]) if w is None else w
    sel = [int(torch.argmin(x @ x.mean(0)))]
    while len(sel) < k:
        d = (1 - x @ x[sel].T).min(1).values.clamp(min=0) * w
        d[sel] = -math.inf
        sel.append(int(torch.argmax(d)))
    return sorted(sel)


def test_budget_rules():
    assert strict_budget(1000) == 50 and strict_budget(100) == 5 and strict_budget(10) == 1
    assert floor_budget(100) == 32 and floor_budget(20) == 20 and floor_budget(10000) == 500
    assert power_budget(10000, scale=1e9) == 10000


def test_divprune_matches_naive():
    g = torch.Generator().manual_seed(0)
    for n, k in ((64, 8), (300, 15), (1000, 50)):
        x = torch.randn(n, 32, generator=g)
        assert max_min_diversity(F.normalize(x, dim=-1), k).tolist() == naive_divprune(x, k)


def test_etd_weighted_matches_naive():
    g = torch.Generator().manual_seed(1)
    x, r = torch.randn(400, 32, generator=g), torch.randn(400, generator=g) * 3
    w = evidence_weights(r, 0.25)
    got = etd_select(x, r, 20, gamma=0.25).tolist()
    assert got == naive_divprune(x, 20, w)


def test_uniform_evidence_is_divprune():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(500, 64, generator=g)
    flat = torch.zeros(500)
    assert torch.allclose(evidence_weights(flat), torch.ones(500))
    assert etd_select(x, flat, 25).tolist() == max_min_diversity(F.normalize(x, dim=-1), 25).tolist()


def test_weights_follow_definition():
    g = torch.Generator().manual_seed(3)
    r = torch.randn(200, generator=g)
    w = evidence_weights(r, 0.25)
    ref = (200 * torch.softmax(r, 0)) ** 0.25
    assert torch.allclose(w, ref / ref.max(), atol=1e-6)


def test_equal_distances_follow_evidence():
    # orthogonal tokens: every distance is 1, so after the centroid seed (index 0) ETD follows the evidence ranking
    g = torch.Generator().manual_seed(4)
    x, r = torch.eye(64), torch.randn(64, generator=g)
    kept = set(etd_select(x, r, 9, gamma=0.25).tolist())
    order = [i for i in torch.argsort(r, descending=True).tolist() if i != 0]
    assert kept == {0, *order[:8]}


def test_scorer_and_loss():
    torch.manual_seed(0)
    net = EvidenceScorer(d=64, h=32)
    gh, gw = 6, 8
    cov, rows, cols = box_coverage(gh, gw, [[0.0, 0.0, 0.25, 0.5]])
    assert cov.shape == (48,) and float(cov.max()) == 1.0
    rel, txt = net(torch.randn(48, 64), torch.randn(5, 64), rows, cols, gh, gw)
    assert rel.shape == (48,) and txt.shape == (48,)
    loss = listwise_loss(rel, cov)
    loss.backward()
    assert loss.item() > 0
    assert 1.8e6 < sum(p.numel() for p in EvidenceScorer().parameters()) < 2.1e6  # ~1.9M at d=2560, h=256


def test_teacher_kl():
    a = torch.randn(4, 10)
    assert abs(teacher_kl(a, a)) < 1e-6 and teacher_kl(a, a.flip(1)) > 0


# ----------------------------------------------------------------------------- glance encoder on a toy tower
class ToyPatchEmbed(nn.Module):
    def __init__(self, c=3, p=2, tp=2, d=16):
        super().__init__()
        self.patch_size, self.temporal_patch_size = p, tp
        self.proj = nn.Linear(c * tp * p * p, d)

    def forward(self, x):
        return self.proj(x.float())


class ToyTower(nn.Module):
    """Same interface as the Qwen3.5 vision tower; blocks are identity so outputs depend on pixels only."""
    spatial_merge_size = 2

    def __init__(self, d=16):
        super().__init__()
        self.patch_embed = ToyPatchEmbed(d=d)
        self.blocks = nn.ModuleList()

    def fast_pos_embed_interpolate(self, grid):
        t, h, w = (int(v) for v in grid[0])
        return torch.zeros(t * h * w, self.patch_embed.proj.out_features)

    def rot_pos_emb(self, grid):
        t, h, w = (int(v) for v in grid[0])
        return torch.zeros(t * h * w, 2)

    def merger(self, x):
        return x.view(-1, 4, x.shape[-1]).mean(1)


def test_glance_output_layout():
    torch.manual_seed(0)
    tower = ToyTower()
    h, w = 16, 12  # patches; merged grid 8 x 6, coarse grid 4 x 3
    patches = torch.randn(h * w, 3 * 2 * 2 * 2)
    out, stats = glance_encode_one(tower, patches, h, w, diversity_picker, 0.25)
    assert out.shape == (h * w // 4, 16)
    assert stats["zoom_cells"] == 3 and stats["pass2_patches"] == 3 * 16 and stats["pass1_patches"] == 48
    gh, gw = h // 2, w // 2
    grid = out.view(gh, gw, -1)
    # every 2x2 block of native slots is either 4 identical coarse copies or 4 second-pass tokens
    copies = 0
    for i in range(0, gh, 2):
        for j in range(0, gw, 2):
            blk = grid[i:i + 2, j:j + 2].reshape(4, -1)
            if torch.allclose(blk, blk[:1].expand_as(blk)):
                copies += 1
    assert copies == 12 - 3
    # with zoom on every cell, the glance output equals the dense output of this toy tower
    full, _ = glance_encode_one(tower, patches, h, w, diversity_picker, 1.0)
    dense = tower.merger(tower.patch_embed(patches))
    assert torch.allclose(full, dense, atol=1e-6)


def test_scorer_picker_is_topk():
    torch.manual_seed(0)
    net = EvidenceScorer(d=16, h=16).eval()
    feats, q = torch.randn(12, 16), torch.randn(3, 16)
    rows, cols = torch.arange(12) // 3, torch.arange(12) % 3
    rel, _ = net(feats, q, rows, cols, 4, 3)
    pick = scorer_picker(net, q)(feats, rows, cols, 4, 3, 3)
    assert set(pick.tolist()) == set(torch.topk(rel, 3).indices.tolist())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
