#!/usr/bin/env python3
"""Train a query-conditioned visual-token scorer on answer/evidence boxes (frozen Qwen3.5-4B vision tower).

Usage (the paper's scorer):
  train_selector.py OUT_DIR --root DATA_ROOT --model Qwen3.5-4B \
      --data DATA_ROOT/data/selector_train --epochs 3 --samples-per-epoch 60000 --general-weight 0.3
DATA_ROOT holds data/benchmarks/{textvqa,textocr} and the build_selector_data.py output; image paths in the records
are relative to it. Without --data it trains on TextOCR-grounded TextVQA train only. With --data it also reports
recall on each source's image-disjoint held-out split. TextVQA val is always reported.

Labels: for each post-merger visual token, the fraction of its cell covered by the answer/evidence boxes, plus
(when the record has OCR word boxes) whether it touches any OCR word. Loss: listwise cross-entropy of the score
softmax against the normalized box coverage, and 0.2 x BCE for "contains text".

Validation compares, at the model's budget K = min(N, max(32, ceil(0.05 N))), how many answer tokens are kept by
  top      top-K by score
  split    top-K/2 by score, the rest by max-min diversity (DivPrune) from that set
  divprune DivPrune, centroid seed (what the DivPrune models use)
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glance_etd.etd import max_min_diversity  # noqa: E402
from glance_etd.scorer import EvidenceScorer as QuerySelector  # noqa: E402,F401  (same parameter names)
from glance_etd.scorer import box_coverage as coverage, sincos  # noqa: E402,F401

ROOT = Path(".")  # set from --root in main()


def _fill(visual, forced, k):
    """Max-min diversity starting from an already selected set."""
    v = visual.float()
    selected = list(forced)
    min_d = 1.0 - (v @ v[forced].T).max(dim=1).values
    min_d[forced] = -torch.inf
    while len(selected) < k:
        p = int(torch.argmax(min_d))
        selected.append(p)
        min_d = torch.minimum(min_d, 1.0 - v @ v[p])
        min_d[p] = -torch.inf
    return torch.sort(torch.tensor(selected, device=visual.device)).values


def budget(n):
    return min(n, max(32, math.ceil(0.05 * n)))


def textvqa_records(questions, boxes, image_root, src):
    """TextVQA questions + build_boxes.py output -> build_selector_data.py records (TextVQA prompt casing)."""
    out = []
    for q in questions:
        b = boxes.get(str(q["question_id"]))
        if b and b["answer_boxes"]:
            out.append({"src": src, "kind": "ocr", "image": str(image_root / f"{q['image_id']}.jpg"),
                        "question": q["question"].capitalize(), "answer": b["answer"],
                        "boxes": b["answer_boxes"], "ocr_boxes": b["ocr_boxes"]})
    return out


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


class Items(torch.utils.data.Dataset):
    def __init__(self, records, processor, tokenizer, pool2=False):
        self.items = records
        self.processor, self.tokenizer = processor, tokenizer
        self.pool2 = pool2  # 2x2 box-pooled image, as the glance pass sees it

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        r = self.items[i]
        try:
            image = Image.open(ROOT / r["image"]).convert("RGB")
        except Exception as exc:
            print(json.dumps({"bad_image": r["image"], "error": repr(exc)}), flush=True)
            return None
        if self.pool2 and min(image.width, image.height) >= 448:  # tiny pooled grids are not supported by the ViT kernel
            image = image.resize((max(1, image.width // 2), max(1, image.height // 2)), Image.BOX)
        px = self.processor(images=image, return_tensors="pt")
        ids = self.tokenizer(r["question"], add_special_tokens=False)["input_ids"]
        return {"src": r["src"], "pixel_values": px["pixel_values"], "grid": px["image_grid_thw"],
                "ids": torch.tensor(ids), "answer_boxes": r["boxes"], "ocr_boxes": r.get("ocr_boxes")}


def features(model, item, device):
    with torch.no_grad():
        out = model.model.get_image_features(item["pixel_values"].to(device), item["grid"].to(device))
        v = out.pooler_output[0]
        q = model.get_input_embeddings()(item["ids"].to(device))
    _, h, w = item["grid"][0].tolist()
    return v, q, h // 2, w // 2


def evaluate(model, selector, data, device, limit):
    selector.eval()
    keys = ("top", "split", "divprune")
    kept = {k: 0 for k in keys}
    hit = {k: 0 for k in keys}
    total = count = 0
    with torch.no_grad():
        for i in range(min(limit, len(data))):
            item = data[i]
            if item is None:
                continue
            v, q, gh, gw = features(model, item, device)
            n = v.shape[0]
            if gh * gw != n:
                continue
            ans, rows, cols = coverage(gh, gw, item["answer_boxes"])
            mask = (ans > 0).to(device)
            if not mask.any():
                continue
            k = budget(n)
            rel, _ = selector(v, q, rows.to(device), cols.to(device), gh, gw)
            visual = F.normalize(v.float(), dim=-1)
            top = torch.topk(rel, k).indices
            sel = {
                "top": top,
                "split": _fill(visual, torch.topk(rel, max(1, k // 2)).indices, k),
                "divprune": max_min_diversity(visual, k, seed="centroid"),
            }
            for name, idx in sel.items():
                m = torch.zeros(n, dtype=torch.bool, device=device)
                m[idx] = True
                kept[name] += int((m & mask).sum())
                hit[name] += int((m & mask).any())
            total += int(mask.sum())
            count += 1
    selector.train()
    return {"questions": count, "answer_tokens": total,
            "recall": {k: round(kept[k] / max(total, 1), 4) for k in keys},
            "any_hit": {k: round(hit[k] / max(count, 1), 4) for k in keys}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--root", type=Path, required=True, help="data root (see module docstring)")
    ap.add_argument("--model", type=Path, required=True, help="Qwen3.5-4B checkpoint directory")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--text-weight", type=float, default=0.2, help="weight of the OCR text BCE (0 = no text head)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=10 ** 9)
    ap.add_argument("--max-val", type=int, default=1500)
    ap.add_argument("--data", type=Path, help="build_selector_data.py output dir; default trains on TextVQA only")
    ap.add_argument("--general-weight", type=float, default=0.3,
                    help="sampling weight of a kind=general record relative to a kind=ocr record")
    ap.add_argument("--samples-per-epoch", type=int, default=0, help="0 = number of training records")
    ap.add_argument("--max-heldout", type=int, default=300, help="per source")
    ap.add_argument("--no-question", action="store_true",
                    help="question-agnostic scorer (query projection frozen at zero; same checkpoint format)")
    ap.add_argument("--pool2", action="store_true", help="train and validate on 2x2-pooled images")
    args = ap.parse_args()
    global ROOT
    ROOT = args.root
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    device = "cuda"
    base = args.model
    processor = AutoProcessor.from_pretrained(base)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)

    bench = ROOT / "data" / "benchmarks"
    image_root = bench / "textvqa" / "train_images"
    val_q = json.load(open(bench / "textvqa" / "TextVQA_0.5.1_val.json"))["data"]
    val_records = textvqa_records(val_q, json.load(open(bench / "textocr" / "textvqa_val_boxes.json")),
                                  image_root, "textvqa-val")
    heldout = {}
    if args.data:
        records = read_jsonl(args.data / "train.jsonl")
        for r in read_jsonl(args.data / "heldout.jsonl"):
            heldout.setdefault(r["src"], []).append(r)
        for src, rows in heldout.items():
            random.Random(0).shuffle(rows)
            heldout[src] = Items(rows[: args.max_heldout], processor.image_processor, processor.tokenizer, args.pool2)
    else:
        train_q = json.load(open(bench / "textocr" / "TextVQA_0.5.1_train.json"))["data"]
        records = textvqa_records(train_q, json.load(open(bench / "textocr" / "textvqa_train_boxes.json")),
                                  image_root, "textocr-textvqa")
    train = Items(records[: args.max_train], processor.image_processor, processor.tokenizer, args.pool2)
    val = Items(val_records, processor.image_processor, processor.tokenizer, args.pool2)
    counts = {}
    for r in train.items:
        counts[r["src"]] = counts.get(r["src"], 0) + 1
    print(json.dumps({"train": len(train), "val": len(val), "train_by_src": counts,
                      "heldout_by_src": {k: len(v) for k, v in heldout.items()}}), flush=True)

    def run_eval(epoch):
        row = {"epoch": epoch, **evaluate(model, selector, val, device, args.max_val)}
        if heldout:
            row["heldout"] = {src: evaluate(model, selector, d, device, args.max_heldout)
                              for src, d in sorted(heldout.items())}
        log.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    per_epoch = args.samples_per_epoch or len(train)
    if args.data:
        weights = [args.general_weight if r["kind"] == "general" else 1.0 for r in train.items]
        sampler = torch.utils.data.WeightedRandomSampler(weights, per_epoch, replacement=True)
    else:
        sampler = torch.utils.data.RandomSampler(train)
    selector = QuerySelector(d=model.config.text_config.hidden_size).to(device)
    if args.no_question:  # every question maps to the same constant vector (the bias), so q carries no information
        with torch.no_grad():
            selector.q[1].weight.zero_()
        selector.q[1].weight.requires_grad_(False)
        selector.q[0].requires_grad_(False)
    opt = torch.optim.AdamW([x for x in selector.parameters() if x.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(per_epoch / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    loader = torch.utils.data.DataLoader(train, batch_size=None, sampler=sampler, num_workers=8,
                                         persistent_workers=True, prefetch_factor=4)
    log = open(args.out / "train.jsonl", "a", buffering=1)
    run_eval(0)
    step, t0, acc = 0, time.time(), []
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad()
        for i, item in enumerate(loader):
            if item is None:
                continue
            v, q, gh, gw = features(model, item, device)
            if gh * gw != v.shape[0]:
                continue
            ans, rows, cols = coverage(gh, gw, item["answer_boxes"])
            rel, txt = selector(v, q, rows.to(device), cols.to(device), gh, gw)
            target = ans.to(device) / ans.sum().clamp_min(1e-6)
            loss = -(target * F.log_softmax(rel, -1)).sum()
            if item["ocr_boxes"] is not None and args.text_weight > 0:
                ocr, _, _ = coverage(gh, gw, item["ocr_boxes"])
                loss = loss + args.text_weight * F.binary_cross_entropy_with_logits(txt, (ocr > 0).float().to(device))
            (loss / args.batch).backward()
            acc.append(loss.item())
            if (i + 1) % args.batch == 0:
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 50 == 0:
                    row = {"epoch": epoch, "step": step, "loss": round(sum(acc) / len(acc), 4),
                           "lr": sched.get_last_lr()[0], "elapsed_s": int(time.time() - t0)}
                    log.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
                    acc = []
        torch.save({"state_dict": selector.state_dict(), "hidden": selector.h,
                    "d": model.config.text_config.hidden_size}, args.out / f"selector_epoch{epoch}.pt")
        run_eval(epoch)
    (args.out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
