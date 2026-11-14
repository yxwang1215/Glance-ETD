#!/usr/bin/env python3
"""Scale a of the continuous water-filling rule r(N) = min(1, a * N^(gamma-1) * c_open), K = max(1, ceil(r N)).

Label-free: uses only the per-image token counts N of a benchmark (and, with --open != 1, whether the prompt asks
for an open-ended answer); no scores. Bisects a so that pooled sum K / sum N equals the target budget.
Input: a jsonl with one line per image, {"n": N, "q": prompt} (any run of glance_etd_runtime.py gives N per image).
Usage: power_scale.py IMAGES.jsonl [--gamma 0.45] [--open 1] --target 0.05
"""
import argparse
import json
import math
import re



def prompt_format(p):
    """Answer format of a prompt; only 'open' receives the --open factor."""
    p = (p or '').lower()
    if re.search(r'yes or no|answer yes|\byes/no\b', p):
        return 'yesno'
    if re.search(r'option|letter|\n[a-d][\.\)]', p):
        return 'mc'
    if 'single word or phrase' in p:
        return 'short'
    return 'open'


def images(path):
    return [(x["n"], x.get("q", "")) for x in map(json.loads, open(path)) if x.get("n")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images')
    ap.add_argument('--gamma', type=float, default=0.45)
    ap.add_argument('--open', type=float, default=1.0)
    ap.add_argument('--target', type=float, default=0.05)
    a = ap.parse_args()
    imgs = images(a.images)
    N = sum(n for n, _ in imgs)
    target = a.target
    cs = [a.open if prompt_format(q[-240:]) == 'open' else 1.0 for _, q in imgs]

    def kn(s):
        return sum(min(n, max(1, math.ceil(min(1.0, s * n ** (a.gamma - 1) * c) * n - 1e-9)))
                   for (n, _), c in zip(imgs, cs)) / N

    lo, hi = 1e-4, 1e3
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if kn(mid) <= target else (lo, mid)
    print(json.dumps({'gamma': a.gamma, 'open': a.open, 'target': target, 'scale': lo, 'kn': kn(lo),
                      'images': len(imgs), 'open_frac': sum(c != 1.0 for c in cs) / len(cs)}))


if __name__ == '__main__':
    main()
