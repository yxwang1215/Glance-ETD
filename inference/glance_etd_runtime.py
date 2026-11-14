#!/usr/bin/env python3
"""Glance-ETD inference on top of the patched LT-OPD code base (see training/README.md).

Installs the final pipeline as runtime patches and then runs the code base's own evaluation entry point
(``evaluation/infer.py``, batch size 1), so prompts, decoding and scoring are those of the LT-OPD evaluation:

  Stage 1  the glance encoder's cell choice uses the evidence scorer: the 25% coarse cells with the highest
           relevance for the question are re-encoded at native resolution (training used max-min diversity).
  Stage 2  the scorer reads the merged native grid and the question embeddings; ETD keeps K tokens with
           argmax d_i (N p_i)^gamma, gamma = 0.25, on top of the DivPrune greedy.
  Budget   strict K = ceil(r N) per image (default r = 0.05), or floor-32, or K ~ N^0.45 (power rule).

Environment
  GLANCE_ETD_SCORER   scorer checkpoint (train with scorer/train_selector.py)              required
  GLANCE_ETD_GAMMA    ETD exponent                                                        0.25
  GLANCE_ETD_STAGE1   scorer | diversity   (diversity = the training-time cell rule)      scorer
  GLANCE_ETD_SELECT   etd | divprune       (divprune = same budget, plain DivPrune)       etd
  GLANCE_ETD_BUDGET   ratio | floor | power                                               ratio
  GLANCE_ETD_RATIO    r of the ratio / floor rules                                        0.05
  GLANCE_ETD_SCALE    a of the power rule  r(N) = min(1, a N^(0.45 - 1))   (power_scale.py)
  LTOPD_SRC           root of the patched LT-OPD code base                                required
Usage: python glance_etd_runtime.py --dataset textvqa --export-dir EXPORT --data-config datasets.json \
           --output-dir OUT [--rank R --world-size W]
"""
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ["LTOPD_SRC"])
sys.path.insert(0, str(SRC))
from lt_opd.cli import _select  # noqa: E402  (activates the learnable_merge variant of the code base)

_select("current")
sys.path.insert(0, str(SRC / "evaluation"))
sys.path.insert(0, str(HERE.parent))

import verl.models.transformers.efficient_vit as ev  # noqa: E402
import verl.models.transformers.qwen3_5 as q35  # noqa: E402
import verl.models.transformers.vision_token_compressor as vtc  # noqa: E402
from glance_etd.etd import evidence_weights, max_min_diversity  # noqa: E402
from glance_etd.glance import diversity_picker, glance_encode_one, scorer_picker  # noqa: E402
from glance_etd.scorer import EvidenceScorer  # noqa: E402

GAMMA = float(os.environ.get("GLANCE_ETD_GAMMA", "0.25"))
STAGE1 = os.environ.get("GLANCE_ETD_STAGE1", "scorer")
SELECT = os.environ.get("GLANCE_ETD_SELECT", "etd")
BUDGET = os.environ.get("GLANCE_ETD_BUDGET", "ratio")
RATIO = float(os.environ.get("GLANCE_ETD_RATIO", "0.05"))
POWER_EXP = 0.45

_SCORER = {"net": None}
CTX = {"model": None, "input_ids": None, "attention_mask": None, "mask": None}
STATE = {"rel": None}
BY_N, ISSUED, VALIDATING = {}, set(), []


def scorer(device):
    if _SCORER["net"] is None:
        _SCORER["net"] = EvidenceScorer.load(os.environ["GLANCE_ETD_SCORER"], device)
    return _SCORER["net"]


# --------------------------------------------------------------------------------------------- question capture
def question_embeds(device):
    """Input embeddings of the question tokens (the route-query mask of the code base), batch size 1."""
    model, ids, mask = CTX["model"], CTX["input_ids"], CTX["mask"]
    if model is None or ids is None or mask is None or mask.shape[0] != 1:
        return None
    qmask = mask[0].to(device=ids.device, dtype=torch.bool)
    if CTX["attention_mask"] is not None:
        qmask = qmask & CTX["attention_mask"][0].to(device=ids.device, dtype=torch.bool)
    if int(qmask.sum()) == 0:
        return None
    return model.get_input_embeddings()(ids[0][qmask]).detach().to(device)


def install_question_capture():
    orig = q35._get_input_embeds

    def wrapped(*args, **kwargs):
        CTX.update(model=args[0], input_ids=args[1],
                   attention_mask=args[2] if len(args) > 2 else kwargs.get("attention_mask"),
                   mask=kwargs.get("compression_query_mask"))
        try:
            return orig(*args, **kwargs)
        finally:
            CTX.update(model=None, input_ids=None, mask=None)

    q35._get_input_embeds = wrapped


# --------------------------------------------------------------------------------------------- Stage 1
def install_stage1():
    def glance_forward_one(visual, patches, h, w, cfg):
        picker = diversity_picker
        if STAGE1 == "scorer":
            q = question_embeds(patches.device)
            if q is not None:
                picker = scorer_picker(scorer(patches.device), q)
        return glance_encode_one(visual, patches, h, w, picker, float(cfg["zoom_ratio"]))

    ev.glance_forward_one = glance_forward_one


# --------------------------------------------------------------------------------------------- budget + Stage 2
def k_for(n):
    if BUDGET == "ratio":
        r, floor = RATIO, 1
    elif BUDGET == "floor":
        r, floor = RATIO, 32
    elif BUDGET == "power":
        r, floor = min(1.0, float(os.environ["GLANCE_ETD_SCALE"]) * n ** (POWER_EXP - 1.0)), 1
    else:
        raise ValueError(f"unknown GLANCE_ETD_BUDGET {BUDGET}")
    return int(min(n, max(floor, math.ceil(r * n - 1e-9))))


def stage2_scores(image_embeds, query_embeds, coords):
    """Scorer logits on the merged native grid; rows / cols come from the M-RoPE coordinates of the image."""
    h = coords[:, 1] - coords[:, 1].min()
    w = coords[:, 2] - coords[:, 2].min()
    gh, gw = int(h.max()) + 1, int(w.max()) + 1
    if gh * gw != image_embeds.shape[0]:
        return None
    dev = image_embeds.device
    with torch.no_grad():
        rel, _ = scorer(dev)(image_embeds.to(dev), query_embeds.to(dev), h.to(dev), w.to(dev), gh, gw)
    return rel.float()


def install_stage2():
    orig_route = vtc.build_cdpruner_route
    orig_budget = vtc._exact_cdpruner_budget
    orig_validate = vtc.DARTMergeRoute.validate
    orig_greedy = vtc._greedy_max_min_diversity

    def budget(num_tokens, retention_bps=500):
        # A route re-validates its own K after generation (multi-image questions may repeat N).
        if VALIDATING and VALIDATING[-1][0] == int(num_tokens):
            return VALIDATING[-1][1]
        k = BY_N.get(int(num_tokens))
        return k if k is not None else orig_budget(num_tokens, retention_bps=retention_bps)

    def validate(self):
        key = (int(self.original_tokens), int(self.output_tokens))
        if key not in ISSUED:
            return orig_validate(self)
        VALIDATING.append(key)
        try:
            return orig_validate(self)
        finally:
            VALIDATING.pop()

    def build_route(image_embeds, *, query_embeds, image_coordinates, **kw):
        n = int(image_embeds.shape[0])
        STATE["rel"] = stage2_scores(image_embeds.detach(), query_embeds.detach(), image_coordinates.detach().long())
        k = k_for(n)
        BY_N[n] = k
        ISSUED.add((n, k))
        return orig_route(image_embeds, query_embeds=query_embeds, image_coordinates=image_coordinates, **kw)

    def greedy(normalized_visual, *, output_tokens, seed="nearest_neighbor", chunk_tokens=2048, **kw):
        rel = STATE["rel"]
        n = normalized_visual.shape[0]
        if SELECT != "etd" or rel is None or rel.numel() != n or output_tokens >= n:
            return orig_greedy(normalized_visual, output_tokens=output_tokens, seed=seed, chunk_tokens=chunk_tokens, **kw)
        kw.pop("importance", None)
        weights = evidence_weights(rel.to(normalized_visual.device), GAMMA)
        return orig_greedy(normalized_visual, output_tokens=output_tokens, seed=seed, chunk_tokens=chunk_tokens,
                           importance=weights, **kw)

    vtc._exact_cdpruner_budget = budget
    vtc.DARTMergeRoute.validate = validate
    vtc.build_cdpruner_route = build_route
    vtc._greedy_max_min_diversity = greedy


def install_validate_tolerance():
    """Learnable-merge routes check per-group prior sums with an FP32 scatter_add at atol 1e-5, which accumulation
    error can exceed for groups of thousands of tokens; on exactly that error, re-check that one test at 1e-4."""
    inner = vtc.DARTMergeRoute.validate
    allclose = torch.allclose
    msg = "Discarded priors must sum to one per non-singleton group and zero per singleton"

    def validate(self):
        try:
            return inner(self)
        except ValueError as e:
            if str(e) != msg:
                raise
        torch.allclose = lambda a, b, rtol=1e-5, atol=1e-8, equal_nan=False: allclose(
            a, b, rtol=rtol, atol=max(atol, 1e-4), equal_nan=equal_nan)
        try:
            return inner(self)
        finally:
            torch.allclose = allclose

    vtc.DARTMergeRoute.validate = validate


def install():
    install_question_capture()
    install_stage1()
    install_stage2()
    install_validate_tolerance()
    print(f"glance-etd: stage1={STAGE1} select={SELECT} gamma={GAMMA} budget={BUDGET} r={RATIO}", flush=True)


if __name__ == "__main__":
    install()
    import infer  # noqa: E402

    for settings in infer.PROTOCOLS.values():
        settings["batch_size"] = 1
    sys.argv = ["infer.py", *sys.argv[1:]]
    infer.main()
