#!/usr/bin/env python3
"""Ground TextVQA answers to TextOCR word boxes.

Usage: build_boxes.py TEXTVQA_JSON TEXTOCR_JSON OUT_JSON

OUT maps question_id -> {image_id, answer, answer_boxes, ocr_boxes}. Boxes are [x0, y0, x1, y1]
normalized by the image size. answer_boxes is empty when no answer (in frequency order) can be spelled
word by word from the image's OCR words; TextOCR marks illegible words as ".", which are skipped.
"""
import collections
import json
import re
import sys

norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def main(textvqa_path, textocr_path, out_path):
    ocr = json.load(open(textocr_path))
    words = collections.defaultdict(list)
    for ann in ocr["anns"].values():
        key = norm(ann["utf8_string"])
        if not key:
            continue
        img = ocr["imgs"][ann["image_id"]]
        x, y, w, h = ann["bbox"]
        box = [x / img["width"], y / img["height"], (x + w) / img["width"], (y + h) / img["height"]]
        words[ann["image_id"]].append((key, [round(min(max(v, 0.0), 1.0), 5) for v in box]))

    out = {}
    stats = collections.Counter()
    for q in json.load(open(textvqa_path))["data"]:
        stats["questions"] += 1
        ws = words.get(q["image_id"])
        if ws is None:
            continue
        stats["with_ocr"] += 1
        answer, boxes = None, []
        for cand, _ in collections.Counter(q["answers"]).most_common():
            toks = [norm(t) for t in cand.split() if norm(t)]
            if toks and all(any(k == t for k, _ in ws) for t in toks):
                answer = cand
                boxes = [b for k, b in ws if k in set(toks)]
                break
        stats["grounded"] += bool(boxes)
        out[str(q["question_id"])] = {
            "image_id": q["image_id"], "answer": answer,
            "answer_boxes": boxes, "ocr_boxes": [b for _, b in ws],
        }
    json.dump(out, open(out_path, "w"))
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main(*sys.argv[1:4])
