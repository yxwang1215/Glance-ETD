#!/usr/bin/env python3
"""Merge box-grounded QA into one jsonl for train_selector.py.

Usage: build_selector_data.py ROOT
ROOT layout: data/benchmarks/textvqa/{TextVQA_0.5.1_val.json, train_images/}, data/benchmarks/textocr/
{TextVQA_0.5.1_train.json, textvqa_train_boxes.json (build_boxes.py)}, data/selector_train/viscot/{metadata,images}
(the Visual-CoT release).

Sources
  textocr-textvqa  TextVQA train answers matched to TextOCR word boxes (build_boxes.py); also has all OCR boxes
  viscot-<name>    Visual-CoT key-region boxes: docvqa, infographicsvqa, dude, sroie, textcap, textvqa (kind=ocr);
                   gqa, visual7w, flickr30k, openimages, vsr (kind=general)

Every image of the TextVQA validation set is excluded (TextCaps and TextVQA share images). Records whose
image cannot be found are dropped and counted. Each source keeps 2% of its images (by hash) as a held-out
split. Output: data/selector_train/{train,heldout}.jsonl and stats.json; boxes are normalized [x0, y0, x1, y1]
and image paths are relative to ROOT.
"""
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path

OCR = {"docvqa", "infographicsvqa", "dude", "sroie", "textcap", "textvqa"}
GENERAL = {"gqa", "visual7w", "flickr30k", "openimages", "vsr"}


def norm_q(q):
    return re.sub(r"\s+", " ", q.strip().lower())


def main(root):
    root = Path(root)
    bench = root / "data" / "benchmarks"
    viscot = root / "data" / "selector_train" / "viscot"
    out = root / "data" / "selector_train"
    tv_images = bench / "textvqa" / "train_images"
    val_images = {q["image_id"] for q in json.load(open(bench / "textvqa" / "TextVQA_0.5.1_val.json"))["data"]}

    index = {}
    for name in sorted(OCR | GENERAL - {"textcap", "textvqa"}):
        d = viscot / "images" / name
        files = {}
        for dirpath, _, names in os.walk(d):
            for n in names:
                if not n.endswith(".part"):
                    files.setdefault(n, os.path.join(dirpath, n))
        index[name] = files

    stats = collections.Counter()
    seen = set()
    records = []

    def add(rec, image_key):
        key = (image_key, norm_q(rec["question"]))
        if key in seen:
            stats[f"{rec['src']}:duplicate"] += 1
            return
        seen.add(key)
        rec["split"] = "heldout" if int(hashlib.md5(image_key.encode()).hexdigest(), 16) % 50 == 0 else "train"
        records.append(rec)
        stats[f"{rec['src']}:{rec['split']}"] += 1

    tv_train = {str(q["question_id"]): q for q in json.load(open(bench / "textocr" / "TextVQA_0.5.1_train.json"))["data"]}
    for qid, b in json.load(open(bench / "textocr" / "textvqa_train_boxes.json")).items():
        q = tv_train.get(qid)
        if q is None or not b["answer_boxes"] or q["image_id"] in val_images:
            continue
        add({"src": "textocr-textvqa", "kind": "ocr", "image": os.path.relpath(tv_images / f"{q['image_id']}.jpg", root),
             "question": q["question"], "answer": b["answer"], "boxes": b["answer_boxes"],
             "ocr_boxes": b["ocr_boxes"]}, f"textvqa/{q['image_id']}")

    for name in sorted(OCR | GENERAL):
        path = viscot / "metadata" / f"{name}_cot_train.jsonl"
        if not path.exists():
            stats[f"viscot-{name}:no_metadata"] += 1
            continue
        for line in open(path):
            r = json.loads(line)
            if not r.get("bboxs"):
                stats[f"viscot-{name}:no_box"] += 1
                continue
            if name in ("textcap", "textvqa"):
                stem = Path(r["image"]).stem
                if stem in val_images:
                    stats[f"viscot-{name}:textvqa_val_image"] += 1
                    continue
                image = tv_images / f"{stem}.jpg"
                if not image.exists():
                    stats[f"viscot-{name}:missing_image"] += 1
                    continue
                image, image_key = os.path.relpath(image, root), f"textvqa/{stem}"
            else:
                image = index[name].get(os.path.basename(r["image"]))
                if image is None:
                    stats[f"viscot-{name}:missing_image"] += 1
                    continue
                image, image_key = os.path.relpath(image, root), f"{name}/{os.path.basename(r['image'])}"
            w, h = float(r["width"]), float(r["height"])
            boxes = [[round(min(max(v, 0.0), 1.0), 5) for v in (b[0] / w, b[1] / h, b[2] / w, b[3] / h)]
                     for b in r["bboxs"]]
            boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
            if not boxes:
                stats[f"viscot-{name}:empty_box"] += 1
                continue
            add({"src": f"viscot-{name}", "kind": "ocr" if name in OCR else "general", "image": image,
                 "question": r["question"], "answer": r.get("answer"), "boxes": boxes, "ocr_boxes": None},
                image_key)

    for split in ("train", "heldout"):
        with open(out / f"{split}.jsonl.partial", "w") as f:
            for rec in records:
                if rec["split"] == split:
                    f.write(json.dumps(rec) + "\n")
        os.replace(out / f"{split}.jsonl.partial", out / f"{split}.jsonl")
    json.dump(dict(sorted(stats.items())), open(out / "stats.json", "w"), indent=1)
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
