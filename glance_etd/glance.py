"""Stage 1: the glance encoder (Sec. 3.2) for a Qwen3.5-VL vision tower.

(i)   Glance: 2x2 average-pool the image in normalised pixel space and encode it with full attention
      (1/4 of the patches); one coarse merged token per 2x2 cell of the native merged grid.
(ii)  Look: choose 25% of the coarse cells -- by max-min diversity during training, by the evidence scorer's
      relevance (top cells) at inference -- and encode the native patches of all chosen cells as one sequence with
      global attention among them, using the absolute and rotary position codes of their native positions.
(iii) Merge: chosen cells take their second-pass tokens; every other cell copies its coarse token to its four
      native slots. The output has the shape and order of the dense encoder's output, so any post-encoder selector
      and the LLM's M-RoPE positions apply unchanged.

Images with fewer than ``min_patches`` ViT patches are encoded densely. The glance encoder adds no parameters.
``visual`` is the HF Qwen3.5 vision tower (patch_embed, fast_pos_embed_interpolate, rot_pos_emb, blocks, merger).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .etd import max_min_diversity

ZOOM_RATIO = 0.25
MIN_PATCHES = 4096

# A cell picker maps (coarse tokens of the candidate cells [M, D], their (row, col) on the coarse grid, the coarse
# grid size (lgh, lgw), how many cells to pick) to indices into the M candidates.
CellPicker = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int], torch.Tensor]


def diversity_picker(feats, rows, cols, lgh, lgw, count):
    """Training-time rule: max-min diversity on the coarse tokens (the MLLM never depends on the scorer)."""
    return max_min_diversity(F.normalize(feats.float(), dim=-1), int(count), seed="centroid")


def scorer_picker(scorer, query_embeds: Optional[torch.Tensor]) -> CellPicker:
    """Inference-time rule: the cells with the highest scorer relevance for the question (no diversity term).
    Falls back to diversity when no question is available."""

    def pick(feats, rows, cols, lgh, lgw, count):
        if query_embeds is None or query_embeds.numel() == 0:
            return diversity_picker(feats, rows, cols, lgh, lgw, count)
        with torch.no_grad():
            rel, _ = scorer(feats, query_embeds.to(feats.device), rows, cols, int(lgh), int(lgw))
        return torch.argsort(rel.float(), descending=True)[:count]

    return pick


# ------------------------------------------------------------------------------------------------ tower helpers
def _rope(visual, grid):
    rot = visual.rot_pos_emb(grid)
    rot = rot.reshape(rot.shape[0], -1)
    emb = torch.cat((rot, rot), dim=-1)
    return emb.cos(), emb.sin()


def _embed(visual, pixel_values, grid):
    hidden = visual.patch_embed(pixel_values) + visual.fast_pos_embed_interpolate(grid)
    return hidden.reshape(hidden.shape[0], -1), _rope(visual, grid)


def _cu(lengths, device):
    return F.pad(torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))


def _run_blocks(visual, hidden, pos, cu):
    for block in visual.blocks:
        hidden = block(hidden, cu_seqlens=cu, position_embeddings=pos)
    return hidden


def _patches_to_image(patches, h, w, p, tp, merge=2):
    c = patches.shape[1] // (tp * p * p)
    x = patches.view(h // merge, w // merge, merge, merge, c, tp, p, p)
    return x.permute(5, 4, 0, 2, 6, 1, 3, 7).reshape(tp, c, h * p, w * p)[0]


def _image_to_patches(image, p, tp, merge=2):
    c, hh, ww = image.shape
    h, w = hh // p, ww // p
    x = image.view(c, h // merge, merge, p, w // merge, merge, p).permute(1, 4, 2, 5, 0, 3, 6)
    x = x.unsqueeze(5).expand(*x.shape[:5], tp, p, p)
    return x.reshape(h * w, c * tp * p * p)


def downsample_patches(patches, h, w, p, tp, merge=2):
    """2x area-average downsample in normalised pixel space; pads to a multiple of 2*merge patches first
    (zero = channel mean after normalisation). Returns (patches, h', w') in processor order."""
    image = _patches_to_image(patches.float(), h, w, p, tp, merge)
    step = 2 * merge
    h4, w4 = math.ceil(h / step) * step, math.ceil(w / step) * step
    image = F.pad(image, (0, (w4 - w) * p, 0, (h4 - h) * p))
    low = F.avg_pool2d(image.unsqueeze(0), 2).squeeze(0)
    return _image_to_patches(low, p, tp, merge).to(patches.dtype), h4 // 2, w4 // 2


def zoom_patch_index(chosen_cells, lgw, gh, gw, merge, device):
    """Coarse cells -> (sorted native merged-unit ids, their patch indices in processor order)."""
    cr, cc = chosen_cells // lgw, chosen_cells % lgw
    da = torch.tensor([0, 0, 1, 1], device=device)
    db = torch.tensor([0, 1, 0, 1], device=device)
    nr, nc = (cr.view(-1, 1) * 2 + da).reshape(-1), (cc.view(-1, 1) * 2 + db).reshape(-1)
    keep = (nr < gh) & (nc < gw)
    native_units = torch.sort((nr * gw + nc)[keep]).values
    unit = merge * merge
    patch_index = (native_units.view(-1, 1) * unit + torch.arange(unit, device=device).view(1, -1)).reshape(-1)
    return native_units, patch_index


# ------------------------------------------------------------------------------------------------ glance encoder
def glance_encode_one(visual, patches, h, w, picker: CellPicker = diversity_picker, zoom_ratio: float = ZOOM_RATIO):
    """One image (t = 1, h x w patches) -> [h/2 * w/2, D] merged tokens in processor order, and stats."""
    merge = visual.spatial_merge_size
    p, tp = visual.patch_embed.patch_size, visual.patch_embed.temporal_patch_size
    device = patches.device
    gh, gw = h // merge, w // merge
    # (i) glance
    low, lh, lw = downsample_patches(patches, h, w, p, tp, merge)
    hidden, pos = _embed(visual, low, torch.tensor([[1, lh, lw]], device=device))
    coarse = visual.merger(_run_blocks(visual, hidden, pos, _cu([lh * lw], device)))
    lgh, lgw = lh // merge, lw // merge
    rows = torch.arange(gh, device=device).view(gh, 1).expand(gh, gw)
    cols = torch.arange(gw, device=device).view(1, gw).expand(gh, gw)
    output = coarse.index_select(0, ((rows // 2) * lgw + cols // 2).reshape(-1))  # coarse token -> 4 native slots
    stats = {"pass1_patches": lh * lw, "pass2_patches": 0, "zoom_cells": 0}
    if zoom_ratio <= 0.0:
        return output, stats
    # (ii) look: candidates are the coarse cells that cover at least one real (unpadded) native cell
    crow = torch.arange(lgh, device=device).view(lgh, 1).expand(lgh, lgw)
    ccol = torch.arange(lgw, device=device).view(1, lgw).expand(lgh, lgw)
    valid = ((crow * 2 < gh) & (ccol * 2 < gw)).reshape(-1).nonzero().squeeze(1)
    count = max(1, min(valid.numel(), round(zoom_ratio * valid.numel())))
    with torch.no_grad():
        local = picker(coarse.detach().index_select(0, valid), valid // lgw, valid % lgw, lgh, lgw, count)
        chosen = torch.sort(valid.index_select(0, local.to(device).long())).values
    native_units, patch_index = zoom_patch_index(chosen, lgw, gh, gw, merge, device)
    grid = torch.tensor([[1, h, w]], device=device)
    pos_embed = visual.fast_pos_embed_interpolate(grid).index_select(0, patch_index)  # native positions
    hidden = visual.patch_embed(patches.index_select(0, patch_index)) + pos_embed
    hidden = hidden.reshape(hidden.shape[0], -1)
    cos, sin = _rope(visual, grid)
    cos, sin = cos.index_select(0, patch_index), sin.index_select(0, patch_index)
    fine = visual.merger(_run_blocks(visual, hidden, (cos, sin), _cu([patch_index.numel()], device)))
    # (iii) merge
    output = output.index_copy(0, native_units, fine.to(output.dtype))
    stats.update(pass2_patches=int(patch_index.numel()), zoom_cells=int(chosen.numel()))
    return output, stats


def glance_visual_forward(visual, pixel_values, grid_thw, picker_for_image=None, zoom_ratio=ZOOM_RATIO,
                          min_patches=MIN_PATCHES):
    """Drop-in for ``visual(pixel_values, grid_thw=grid_thw).pooler_output``.

    Images with t == 1 and at least ``min_patches`` patches take the glance path; the rest (small images, videos)
    are encoded densely. ``picker_for_image(i)`` returns the cell picker for image i (default: diversity)."""
    rows = [tuple(int(v) for v in r) for r in grid_thw.tolist()]
    sizes = [t * h * w for t, h, w in rows]
    merge = visual.spatial_merge_size
    big = [i for i, (t, h, w) in enumerate(rows) if t == 1 and t * h * w >= min_patches]
    if not big:
        return visual(pixel_values, grid_thw=grid_thw).pooler_output
    starts = [0]
    for n in sizes:
        starts.append(starts[-1] + n)
    outputs = [None] * len(rows)
    small = [i for i in range(len(rows)) if i not in big]
    if small:
        idx = torch.cat([torch.arange(starts[i], starts[i + 1], device=pixel_values.device) for i in small])
        dense = visual(pixel_values.index_select(0, idx), grid_thw=grid_thw[small]).pooler_output
        offset = 0
        for i in small:
            n = sizes[i] // (merge * merge)
            outputs[i] = dense[offset: offset + n]
            offset += n
    for i in big:
        _, h, w = rows[i]
        picker = picker_for_image(i) if picker_for_image is not None else diversity_picker
        outputs[i], _ = glance_encode_one(visual, pixel_values[starts[i]: starts[i + 1]], h, w, picker, zoom_ratio)
    return torch.cat(outputs, dim=0)
