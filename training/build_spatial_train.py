#!/usr/bin/env python3
"""Build LT-OPD-14K-spatial: 12,500 LT-14K rows (proportional per bucket) + 1,500 spatial MCQ rows.

Spatial rows: 750 SAT static (synthetic ProcTHOR: 250 counting / 250 depth / 250 relation) and 750 generated from
COCO train2017 instance annotations (375 counting / 375 left-right or above-below), in the CV-Bench question style.
COCO images used by GQA testdev (via Visual Genome coco_id), POPE (val2014 ids), MME and CV-Bench are excluded.
Rows use the LT-14K MCQ protocol, route-query builder and visual capacity profiles; both are first validated by
reproducing existing LT-14K rows exactly.

Usage: LTOPD_DATA_ROOT=DIR LTOPD_SRC=PATCHED_LTOPD_SRC python build_spatial_train.py
DIR holds data/LT-OPD-14K, data/spatial_train (SAT static + COCO instances), data/benchmarks/* (for exclusion)
and models/Qwen3.5-4B (processor only).
"""
import hashlib, io, json, os, random, re, sys, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ['LTOPD_DATA_ROOT'])
SRC = Path(os.environ['LTOPD_SRC'])
sys.path[:0] = [str(SRC / 'learnable_merge'), str(SRC)]
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

BASE = ROOT / 'data' / 'LT-OPD-14K'
RAW = ROOT / 'data' / 'spatial_train'
OUT = ROOT / 'data' / 'LT-OPD-14K-spatial'
MEDIA = OUT / 'media'
SEED = 'lt14k-spatial-20261030'
QUOTAS_BASE = {'onethinker_mcq': 3000, 'onethinker_math': 1500, 'onethinker_numerical': 900, 'onethinker_ocr': 900,
               'onethinker_regression': 900, 'pixmo_ask_model_anything': 3000, 'llava_v1_5_mix665k': 1800,
               'textvqa': 1000, 'vision_opd_fullimage_mcq': 1000}
SPATIAL_ROWS = 1500


def key(*parts):
    return hashlib.sha256('|'.join([SEED, *map(str, parts)]).encode()).hexdigest()


def spatial_quotas():
    """Remove 1,500 rows from the nine buckets in proportion (largest remainder), keeping 14,000 total."""
    total = sum(QUOTAS_BASE.values())
    keep_total = total - SPATIAL_ROWS
    exact = {b: q * keep_total / total for b, q in QUOTAS_BASE.items()}
    quotas = {b: int(v) for b, v in exact.items()}
    for b in sorted(exact, key=lambda b: (-(exact[b] - quotas[b]), b))[: keep_total - sum(quotas.values())]:
        quotas[b] += 1
    quotas['spatial'] = SPATIAL_ROWS
    assert sum(quotas.values()) == total
    return quotas


# ---------------------------------------------------------------- benchmark exclusions
def excluded_coco_ids():
    ids = set()
    vg = {d['image_id']: d['coco_id'] for d in json.loads((RAW / 'vg' / 'image_data.json').read_text()) if d.get('coco_id')}
    gqa = pq.read_table(ROOT / 'data/benchmarks/gqa/testdev_balanced_instructions/testdev-00000-of-00001.parquet',
                        columns=['imageId']).column(0).to_pylist()
    gqa_coco = {vg[int(i)] for i in set(gqa) if str(i).isdigit() and int(i) in vg}
    from datasets import load_from_disk
    pope = load_from_disk(str(ROOT / 'data/benchmarks/pope/dataset'))['test']
    pope_ids = {int(m) for s in pope['image_source'] for m in re.findall(r'(\d{6,12})', str(s))}
    mme = load_from_disk(str(ROOT / 'data/benchmarks/mme/dataset'))['test']
    mme_ids = {int(m) for s in mme['question_id'] for m in re.findall(r'(\d{9,12})', str(s))}
    cv = set()
    for name in ('test_2d.parquet', 'test_3d.parquet'):
        for s in pq.read_table(ROOT / 'data/benchmarks/spatial_raw/cvbench' / name, columns=['source_filename']).column(0).to_pylist():
            cv.update(int(m) for m in re.findall(r'(\d{9,12})', str(s)))
    ids = gqa_coco | pope_ids | mme_ids | cv
    print(f'excluded COCO ids: gqa {len(gqa_coco)} pope {len(pope_ids)} mme {len(mme_ids)} cvbench {len(cv)} -> {len(ids)}', flush=True)
    return ids


# ---------------------------------------------------------------- question sources
def mcq(question, choices, correct, rng):
    choices = list(dict.fromkeys(choices))
    rng.shuffle(choices)
    return question, choices, 'ABCDEFGH'[choices.index(correct)]


def sat_items():
    kinds = {'count': re.compile(r'^how many', re.I), 'depth': re.compile(r'closer to the camera|farther from the camera', re.I),
             'relation': re.compile(r'\b(left|right|above|below|behind|in front of)\b', re.I)}
    pools = defaultdict(list)
    f = pq.ParquetFile(RAW / 'sat' / 'SAT_static.parquet')
    index = 0
    for batch in f.iter_batches(batch_size=2048, columns=['question', 'answers', 'correct_answer']):
        for row in batch.to_pylist():
            kind = next((k for k, p in kinds.items() if p.search(row['question'])), None)
            if kind and 2 <= len(row['answers']) <= 6 and row['correct_answer'] in row['answers']:
                pools[kind].append((key('sat', index), index, row))
            index += 1
    picked = {k: sorted(v)[:250] for k, v in pools.items()}
    print('SAT pools', {k: len(v) for k, v in pools.items()}, flush=True)
    wanted = {i for v in picked.values() for _, i, _ in v}
    images = {}
    index = 0
    for batch in f.iter_batches(batch_size=512, columns=['image_bytes']):
        for value in batch.column(0).to_pylist():
            if index in wanted:
                images[index] = value[0]
            index += 1
    items = []
    for kind, rows in picked.items():
        for k, i, row in rows:
            rng = random.Random(k)
            q, choices, letter = mcq(row['question'].strip(), row['answers'], row['correct_answer'], rng)
            items.append({'uid_key': f'sat:{i}', 'kind': f'sat_{kind}', 'question': q, 'choices': choices,
                          'answer': letter, 'image_bytes': images[i]})
    return items


def coco_items(excluded):
    data = json.loads((RAW / 'coco' / 'annotations' / 'instances_train2017.json').read_text())
    names = {c['id']: c['name'] for c in data['categories']}
    images = {im['id']: im for im in data['images'] if im['id'] not in excluded}
    by_image = defaultdict(list)
    for a in data['annotations']:
        if a['image_id'] in images:
            by_image[a['image_id']].append(a)
    counts, relations = [], []
    for image_id, anns in by_image.items():
        im = images[image_id]
        area = im['width'] * im['height']
        per_cat = defaultdict(list)
        for a in anns:
            per_cat[a['category_id']].append(a)
        for cat, group in per_cat.items():
            # Countable: no crowd regions and every instance clearly visible (>= 0.5% of the image).
            if any(a['iscrowd'] for a in group) or any(a['area'] < 0.005 * area for a in group):
                continue
            if 1 <= len(group) <= 6:
                counts.append((key('coco-count', image_id, cat), image_id, cat, len(group)))
        singles = [(cat, g[0]) for cat, g in per_cat.items()
                   if len(g) == 1 and not g[0]['iscrowd'] and g[0]['area'] >= 0.01 * area]
        for (ca, a), (cb, b) in [(x, y) for x in singles for y in singles if x[0] < y[0]]:
            ax0, ay0, aw, ah = a['bbox']; bx0, by0, bw, bh = b['bbox']
            if ax0 + aw <= bx0 or bx0 + bw <= ax0:      # disjoint horizontally
                relation = 'left' if ax0 + aw <= bx0 else 'right'
                relations.append((key('coco-rel', image_id, ca, cb), image_id, ca, cb, relation, ('left', 'right')))
            elif ay0 + ah <= by0 or by0 + bh <= ay0:    # disjoint vertically
                relation = 'above' if ay0 + ah <= by0 else 'below'
                relations.append((key('coco-rel', image_id, ca, cb), image_id, ca, cb, relation, ('above', 'below')))
    print('COCO pools', {'count': len(counts), 'relation': len(relations)}, flush=True)
    items, used_images = [], set()
    for k, image_id, cat, n in sorted(counts):
        if len([x for x in items if x['kind'] == 'coco_count']) == 375:
            break
        if image_id in used_images:
            continue
        rng = random.Random(k)
        distractors = rng.sample([str(x) for x in range(0, 8) if x != n], 3)
        q, choices, letter = mcq(f'How many {names[cat]}s are in the image?', [str(n), *distractors], str(n), rng)
        used_images.add(image_id)
        items.append({'uid_key': f'coco-count:{image_id}:{cat}', 'kind': 'coco_count', 'question': q, 'choices': choices,
                      'answer': letter, 'coco_id': image_id})
    for k, image_id, ca, cb, relation, pair in sorted(relations):
        if len([x for x in items if x['kind'] == 'coco_relation']) == 375:
            break
        if image_id in used_images:
            continue
        rng = random.Random(k)
        question = (f'Considering the relative positions of the {names[ca]} and the {names[cb]} in the image, '
                    f'where is the {names[ca]} located with respect to the {names[cb]}?')
        q, choices, letter = mcq(question, list(pair), relation, rng)
        used_images.add(image_id)
        items.append({'uid_key': f'coco-rel:{image_id}:{ca}:{cb}', 'kind': 'coco_relation', 'question': q,
                      'choices': choices, 'answer': letter, 'coco_id': image_id})
    return items


# ---------------------------------------------------------------- LT-14K row format
def protocol_suffix(base_rows):
    suffixes = set()
    for row in base_rows:
        if row['sampling_bucket'] == 'onethinker_mcq':
            content = row['prompt'][0]['content']
            suffixes.add(content[content.index('\n\nResponse requirements:'):])
    assert len(suffixes) == 1, f'MCQ protocol suffix is not unique: {len(suffixes)}'
    return suffixes.pop()


def make_prompt(question, options, suffix):
    return '<image>\n' + question + '\nOptions:\n' + '\n'.join(options) + suffix


class Capacity:
    """Reproduce LT-14K visual_capacity_profiles; the template is chosen by exact match on existing rows."""

    PROFILES = (2097152, 4194304, 8388608, 16777216)

    def __init__(self):
        from transformers import AutoProcessor
        from verl.models.transformers.visual_token_curriculum import visual_token_budget
        self.budget = visual_token_budget
        self.processor = AutoProcessor.from_pretrained(ROOT / 'models' / 'Qwen3.5-4B')
        self.templates = {
            'base': self.processor.chat_template,
            'answer_prefill': (SRC / 'learnable_merge' / 'training' / 'perception_chat_template_qwen35_answer_prefill.jinja').read_text(),
        }
        self.template = None

    def profile(self, image, content, template):
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                 {'type': 'text', 'text': content.replace('<image>', '', 1)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  chat_template=self.templates[template])
        result = {}
        for max_pixels in self.PROFILES:
            image_processor = self.processor.image_processor
            image_processor.max_pixels = max_pixels
            image_processor.size = {**dict(image_processor.size), 'longest_edge': max_pixels}
            encoded = self.processor(text=[text], images=[image], return_tensors='pt')
            grid = encoded['image_grid_thw'][0]
            raw = int(grid.prod())
            dense = raw // 4
            teacher = int(encoded['input_ids'].shape[1])
            budget = int(self.budget(dense, retention_bps=500, minimum_tokens=32))
            result[str(max_pixels)] = {'max_pixels': max_pixels, 'dense_visual_tokens': dense,
                                       'dense_teacher_prompt_tokens': teacher,
                                       'merged_student_prompt_tokens': teacher - dense + budget,
                                       'raw_patch_tokens_per_view': raw}
        return result

    def calibrate(self, base_rows):
        sample = [r for r in base_rows if r['sampling_bucket'] in ('onethinker_mcq', 'vision_opd_fullimage_mcq')][:12]
        for template in self.templates:
            ok = True
            for row in sample:
                image = Image.open(row['images'][0]['path']).convert('RGB')
                got = self.profile(image, row['prompt'][0]['content'], template)
                want = {k: v for k, v in row['extra_info']['visual_capacity_profiles'].items() if v}
                if got != want:
                    ok = False
                    print(f'calibration {template}: mismatch on {row["sample_uid"][:12]}', json.dumps(got)[:300], '!=', json.dumps(want)[:300], flush=True)
                    break
            if ok:
                self.template = template
                print(f'calibration ok with {template} template on {len(sample)} rows', flush=True)
                return
        raise SystemExit('capacity profiles could not be reproduced; refusing to build rows')


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    from verl.utils.route_query import build_general_route_query
    base_table = pq.read_table(BASE / 'train.parquet')
    base_rows = base_table.to_pylist()
    suffix = protocol_suffix(base_rows)
    for row in base_rows[:200]:
        if row['task_type'] == 'mcq':
            rq = build_general_route_query(row['question'], row['options']).as_data_field()
            assert rq == row['route_query'], ('route query drift', row['sample_uid'])
            assert row['prompt'][0]['content'] == make_prompt(row['question'], row['options'], suffix) or \
                row['sampling_bucket'] != 'onethinker_mcq', ('prompt drift', row['sample_uid'])
    print('route query and MCQ prompt reproduce LT-14K rows', flush=True)
    capacity = Capacity()
    capacity.calibrate(base_rows)

    items = sat_items() + coco_items(excluded_coco_ids())
    assert len(items) == SPATIAL_ROWS, Counter(x['kind'] for x in items)
    print('spatial items', Counter(x['kind'] for x in items), flush=True)
    rows = []
    for index, item in enumerate(items):
        if 'coco_id' in item:
            path = MEDIA / f"coco_train2017_{item['coco_id']:012d}.jpg"
            if not path.exists():
                url = f"http://images.cocodataset.org/train2017/{item['coco_id']:012d}.jpg"
                with urllib.request.urlopen(url, timeout=120) as response:
                    payload = response.read()
                path.with_suffix('.part').write_bytes(payload)
                path.with_suffix('.part').rename(path)
        else:
            digest = hashlib.sha256(item['image_bytes']).hexdigest()
            path = MEDIA / f'sat_{digest}.png'
            if not path.exists():
                Image.open(io.BytesIO(item['image_bytes'])).convert('RGB').save(path)
        image = Image.open(path).convert('RGB')
        options = [f'{letter}) {choice}' for letter, choice in zip('ABCDEFGH', item['choices'])]
        content = make_prompt(item['question'], options, suffix)
        rows.append({
            'sample_uid': hashlib.sha256(f"lt14k-spatial|{item['uid_key']}".encode()).hexdigest(),
            'data_source': 'spatial', 'sampling_bucket': 'spatial', 'task_type': 'mcq',
            'question': item['question'], 'options': options, 'answer': item['answer'],
            'prompt': [{'content': content, 'role': 'user'}],
            'images': [{'path': str(path)}], 'teacher_images': [{'path': str(path)}],
            'route_query': build_general_route_query(item['question'], options).as_data_field(),
            'extra_info': {'index': -1, 'visual_capacity_profiles': capacity.profile(image, content, capacity.template)},
            'spatial_kind': item['kind'],
        })
        if (index + 1) % 250 == 0:
            print(f'built {index + 1}/{len(items)}', flush=True)

    quotas = spatial_quotas()
    keep = []
    by_bucket = defaultdict(list)
    for row in base_rows:
        by_bucket[row['sampling_bucket']].append(row)
    for bucket, quota in quotas.items():
        if bucket == 'spatial':
            continue
        keep += sorted(by_bucket[bucket], key=lambda r: key('keep', r['sample_uid']))[:quota]
    kinds = {r['sample_uid']: r.pop('spatial_kind') for r in rows}
    merged = keep + rows
    for i, row in enumerate(merged):
        row['extra_info']['index'] = i
    assert len(merged) == 14000 and len({r['sample_uid'] for r in merged}) == 14000
    table = pa.Table.from_pylist(merged, schema=base_table.schema)
    tmp = OUT / 'train.parquet.part'
    pq.write_table(table, tmp, compression='zstd')
    tmp.replace(OUT / 'train.parquet')
    manifest = {'rows': 14000, 'quotas': quotas, 'spatial_kinds': Counter(kinds.values()),
                'base_sha256': hashlib.sha256((BASE / 'train.parquet').read_bytes()).hexdigest(),
                'train_sha256': hashlib.sha256((OUT / 'train.parquet').read_bytes()).hexdigest(),
                'capacity_template': capacity.template, 'seed': SEED}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('manifest', json.dumps(manifest), flush=True)
    print('SPATIAL_TRAIN_OK')


if __name__ == '__main__':
    main()
