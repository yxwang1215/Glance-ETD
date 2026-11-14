<div align="center">

# Glance, then Select:<br>Evidence-Tempered Visual Token Selection<br>for Extremely Compressed Multimodal LLMs

[![Project](https://img.shields.io/badge/Project-Page-0A7CFF?style=flat-square&logo=googlechrome&logoColor=white)](https://yxwang1215.top/papers/glance-etd/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square)](https://github.com/yxwang1215/Glance-ETD/blob/main/assets/paper.pdf)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/yxwang1215/Glance-ETD)
[![CVPR](https://img.shields.io/badge/CVPR-2027_Submission-1f6feb?style=flat-square)](https://yxwang1215.top/papers/glance-etd/)

<br>

Yuxuan Wang<sup>1</sup> · Guangxiang Zhao<sup>2</sup> · Yuxuan Tian<sup>1</sup> · Yahui Han<sup>1</sup> · Xinyu Tan<sup>1</sup><br>
Feiyu Wang<sup>1</sup> · Zongwei Lv<sup>1</sup> · Xiangzheng Zhang<sup>2</sup> · Lin Sun<sup>2</sup> · Tong Yang<sup>1</sup>

<sup>1</sup><a href="https://cs.pku.edu.cn/">Peking University</a> &nbsp;&nbsp; <sup>2</sup>Qiyuan Tech

<br>

<a href="https://yxwang1215.top/papers/glance-etd/"><b>Project Page</b></a> &nbsp;|&nbsp;
<a href="assets/paper.pdf"><b>Paper</b></a> &nbsp;|&nbsp;
<a href="https://github.com/yxwang1215/Glance-ETD"><b>Code</b></a>

<br>

<a href="#abstract">Abstract</a> &nbsp;·&nbsp;
<a href="#method">Method</a> &nbsp;·&nbsp;
<a href="#results">Results</a> &nbsp;·&nbsp;
<a href="#quick-check">Quick Check</a> &nbsp;·&nbsp;
<a href="#reproducing-the-experiments">Reproduce</a> &nbsp;·&nbsp;
<a href="#related-code-and-papers">Related</a> &nbsp;·&nbsp;
<a href="#citation">Citation</a>

<br>

<img src="assets/overview.png" width="100%" alt="Glance-ETD overview. Training distills a glance-encoder student that selects with DivPrune. At inference the scorer keeps the top 25 percent of coarse cells for native re-encoding, then ETD with exponent 0.25 keeps a strict five percent of tokens.">

*Training and inference. The student is distilled toward a frozen full-token teacher while it selects with DivPrune. The scorer is trained separately on box targets. At inference it picks the top 25% of coarse cells, and ETD keeps K = ⌈0.05 N⌉ tokens with γ = 0.25.*

</div>

## News

- **2026-11-14:** Code, [project page](https://yxwang1215.top/papers/glance-etd/), and [PDF](assets/paper.pdf) are released. Checkpoints are not included.

## Abstract

A high-resolution image becomes thousands of visual tokens, and those tokens dominate prefill. At a strict 5% budget the language model sees about one token in twenty, but two costs remain: the vision encoder still processes every patch, and the few surviving tokens can miss the region that holds the answer. Glance-ETD uses one small question-conditioned scorer for both decisions. A glance encoder reads a low-resolution view and re-encodes only the question-relevant regions at native resolution. [Evidence-Tempered Diversity](#stage-2-etd) then keeps the final tokens by combining [DivPrune](https://arxiv.org/abs/2503.02175)'s max–min distance with the scorer's evidence map.

With 5% of the visual tokens, Glance-ETD retains **91.6%** of the full-token performance of [Qwen3.5-4B](https://qwen.ai/blog?id=qwen3.5) across fourteen benchmarks. That is 17.3 points above the strongest prior method and 6.0 points above the same training recipe with DivPrune. On 4K images it shortens time to first token by **3.1×**. The same comparison holds on [Gemma 4 E4B](https://arxiv.org/abs/2607.02770), GLM-4.6V-Flash, and InternVL3.5-8B, with the largest gains on text-rich tasks.

## Method

One scorer, about 1.9M parameters, reads the visual tokens and the question and outputs a relevance logit for every visual token. It is trained once, with the MLLM frozen, by a listwise loss on box-coverage targets. TextVQA answers are grounded on TextOCR word boxes and merged with Visual-CoT boxes. The same scorer is shared by every Qwen checkpoint.

**Stage 1, glance.** The image is average-pooled 2×2 and encoded at a quarter of the patches. The scorer keeps the 25% of coarse cells with the highest question relevance. Only those cells are re-encoded at native resolution; every other cell copies its coarse token onto the native grid. The grid still has all N positions, so any post-encoder selector applies unchanged, and the encoder costs about a sixth of a dense ViT. Images below 4096 patches are encoded densely. During training the cells are chosen by max–min diversity, so the MLLM does not depend on the scorer. At inference the scorer chooses the cells.

<a id="stage-2-etd"></a>

**Stage 2, ETD.** [DivPrune](https://arxiv.org/abs/2503.02175) starts from the token farthest from the mean and greedily adds

```text
i* = argmax_{i not in S}  d_i,    d_i = min_{j in S} (1 - cos(x_i, x_j))
```

until `|S| = K`. ETD keeps that loop and multiplies the distance by the relative evidence:

```text
i* = argmax_{i not in S}  d_i · (N p_i)^γ,    γ = 0.25
```

`p` is a softmax over the scorer logits. A flat map returns DivPrune. A peaked map moves the selection toward the answer without dropping the rest of the scene. The strict budget is `K = ⌈0.05 N⌉` on every image. Floor-32, `K = max(32, ⌈0.05 N⌉)`, and the label-free rule `K ∝ N^0.45` are the other budgets in the paper. Kept tokens go to the LLM with their original M-RoPE positions.

**Training.** The MLLM and the glance encoder are trained together by on-policy self-distillation for 175 steps. The student rolls out with DivPrune at 5% and is distilled token by token toward a frozen copy of itself that sees every token. The vision-tower learning rate is 2e-6. The scorer is not updated in this stage. The student path is the [patch](training/ltopd_glance_etd.patch) in this repository.

<img src="assets/motivation.png" width="100%" alt="Motivation at a strict five percent budget. Time to first token on HR-Bench 4K, one TextVQA example, and the share of answer tokens kept by each selector.">

*Two limits at a strict 5% budget on Qwen3.5-4B. Left: on HR-Bench 4K, pruning after the vision encoder still spends 2.59 s in the encoder; the glance encoder cuts that stage to 0.65 s. Middle: on one TextVQA image, DivPrune keeps 1 of 10 text tokens, relevance top-K keeps all 10 but little context, and ETD keeps 4 plus context. Right: on 4,123 TextVQA validation questions, bars are the share of tokens that overlap the answer box and diamonds are the share of questions that keep at least one.*

## Results

Two averages appear below, and they are not the same number. **91.6%** is the mean retention over fourteen benchmarks (the nine in the budget plot, plus [V\*](https://arxiv.org/abs/2312.14135), HR-Bench 4K/8K, DocVQA, and InfoVQA). **93.3%** is the mean over those nine benchmarks only. Retention is each score divided by that model's own non-thinking full-token score.

| Setting | Retention | Comparison |
|---|---:|---|
| Glance-ETD, 14 benchmarks, strict 5% | 91.6 ± 0.3% | +17.3 vs. the strongest prior method, +6.0 vs. the same recipe with [DivPrune](https://arxiv.org/abs/2503.02175) |
| Glance-ETD, 9 benchmarks, strict 5% | 93.3% | +17.5 vs. DivPrune; mean of three training seeds |
| Frozen MLLM + ETD, 14 benchmarks | 82.0% | +10.7 vs. DivPrune, the strongest training-free baseline |
| Same recipe, DivPrune, 14 benchmarks | 85.6% | ETD on that recipe reaches 90.5% before the glance encoder |
| HR-Bench 4K time to first token | 3.09× | ViT time 2.60 s → 0.57 s; prefill 482 → 65 TFLOPs |

Inside the 10× learning-rate recipe, ETD adds 3.7 points on the dense encoder (92.5% on the nine-benchmark mean) and the glance encoder adds 0.8 more (93.3%) while cutting HR-Bench 4K prefill from 373 to 65 TFLOPs and time to first token from 3318 to 1283 ms. Setting γ = 0 makes ETD identical to DivPrune and costs 2.9 points, mostly on reading. Setting γ = 64 keeps only the highest-evidence tokens and costs 3.6 points, because the same words are kept several times and the rest of the scene is missed. On TextVQA, ETD keeps a token inside the answer box for 99.1% of questions, against 80.2% for DivPrune.

### Budget

Solid lines are trained models. Dashed lines are training-free and run on the released model. At 5% of the tokens, Glance-ETD is the mean of three seeds; the other budgets use one seed.

<p align="center">
  <img src="assets/budget.png" width="78%" alt="Retention of Qwen3.5-4B from 5 to 50 percent of visual tokens. Glance-ETD is the top curve and stays near the full-token line.">
</p>

*Nine-benchmark retention on Qwen3.5-4B. At 5%, Glance-ETD keeps 93.3% of the full-token score, 17.5 points above [DivPrune](https://arxiv.org/abs/2503.02175).*

### Efficiency

Measured on one A800 in bf16. Glance-ETD is the point that keeps high retention and a large speed-up on HR-Bench 4K. Selectors that prune only after a dense encoder stay near 1×, because the encoder is still the bottleneck. Batch-8 peak memory on HR-Bench 4K drops from 34.7 GB to 17.7 GB, and batch-8 throughput is 2.94× higher. End-to-end time on HR-Bench 4K is 2.69× shorter. On the six benchmarks whose images fall below the 4096-patch glance threshold, time to first token is 0.87–0.90× the full-token run.

<p align="center">
  <img src="assets/efficiency.png" width="100%" alt="Efficiency at a strict five percent budget: retention against HR-Bench 4K speed-up, peak memory, prefill speed-up, and end-to-end speed-up for Qwen3.5-4B and Gemma 4 E4B.">
</p>

*Strict 5%. (a) Nine-benchmark retention against HR-Bench 4K time-to-first-token speed-up. (b) HR-Bench 4K peak memory at batch 1, 4, and 8. (c)(d) Prefill and end-to-end speed-up on the benchmarks with the largest gain. Token counts under the labels are the mean visual tokens of the full model.*

<p align="center">
  <img src="assets/efficiency-supp.png" width="100%" alt="Supplementary efficiency plots for Qwen3.5-4B at a strict five percent budget, including task-matched speed, prefill FLOPs, stage breakdown, throughput, memory, and decode cache.">
</p>

*More of the same measurements: task-matched accuracy against speed on HR-Bench 4K and [V\*](https://arxiv.org/abs/2312.14135), prefill FLOPs as the budget goes from 1% to 25%, a stage breakdown, throughput, peak memory, and the decode cache.*

### Other models

Each model has its own scorer, trained on the same box data from that model's features, and is self-distilled with the same recipe. ETD leads DivPrune on every model, by 8.0–13.0 points when the MLLM is frozen and by 3.7–12.0 points after self-distillation. Glance-ETD stays above self-distillation with DivPrune on all four, and within 2.4 points of the dense encoder.

<p align="center">
  <img src="assets/backbones.png" width="100%" alt="Strict five percent retention on Qwen3.5-4B, Gemma 4 E4B, GLM-4.6V-Flash, and InternVL3.5-8B for DivPrune, ETD, self-distillation, and Glance-ETD.">
</p>

*Nine-benchmark retention at strict 5%. SD is self-distillation with the dense encoder. Glance-ETD is 93.3% on [Qwen3.5-4B](https://qwen.ai/blog?id=qwen3.5), 91.1% on [Gemma 4 E4B](https://arxiv.org/abs/2607.02770), 88.2% on GLM-4.6V-Flash, and 83.8% on InternVL3.5-8B.*

On Gemma 4 E4B-it the same gap shows up across budgets. With the MLLM frozen, ETD keeps 78.6% at strict 5% on the fourteen-benchmark suite, 11.8 points above DivPrune. On the nine-benchmark curve below, ETD at 10% matches DivPrune at 25%.

<p align="center">
  <img src="assets/gemma.png" width="72%" alt="Retention of Gemma 4 E4B-it across visual-token budgets for ETD and DivPrune. ETD is above DivPrune at 5, 10, and 25 percent.">
</p>

*Gemma 4 E4B-it against its own full-token scores. Every marker averages the nine benchmarks, and both methods pass the same number of tokens per image.*

## Repository

| Path | Contents |
|---|---|
| [`glance_etd/etd.py`](glance_etd/etd.py) | DivPrune greedy, ETD weights `(N p)^γ`, strict / floor-32 / `K ∝ N^0.45` budgets. |
| [`glance_etd/scorer.py`](glance_etd/scorer.py) | Evidence scorer, box-coverage targets, listwise loss. |
| [`glance_etd/glance.py`](glance_etd/glance.py) | Two-pass glance encoder. Training uses diversity cells; inference uses the scorer's top cells. |
| [`glance_etd/teacher_kl.py`](glance_etd/teacher_kl.py) | Teacher-KL cost of a kept subset, the diagnostic used to choose γ. |
| [`scorer/`](scorer/) | TextVQA/TextOCR grounding, Visual-CoT merge, and scorer training. |
| [`training/ltopd_glance_etd.patch`](training/ltopd_glance_etd.patch) | Training-code patch: glance encoder on the student path, and the training-side ETD option. |
| [`training/configs/`](training/configs/) | Final recipe and its seeds, the dense-encoder variant, the learning-rate ablations, and ETD-trained glance. |
| [`training/build_spatial_train.py`](training/build_spatial_train.py) | Builds the spatial training mixture used by the recipes. |
| [`inference/`](inference/) | Full inference pipeline and the label-free scale of `K ∝ N^0.45`. |
| [`tests/`](tests/) | CPU checks of the reference implementation. |
| [`assets/`](assets/) | Figures and the [PDF](assets/paper.pdf). |

`glance_etd/` needs only PyTorch. The patch, `scorer/`, and `inference/` reproduce the experiments. The reference functions were checked on CPU against the code used for the paper: DivPrune and ETD selection, the scorer forward pass, and the glance encoder under both cell rules. Model checkpoints are not in this repository.

## Quick check

Tested with PyTorch 2.6. The tests compare ETD with a brute-force reference, check that uniform evidence returns DivPrune, and check the glance output layout on a toy tower.

```bash
git clone https://github.com/yxwang1215/Glance-ETD.git
cd Glance-ETD
pip install torch
python tests/test_reference.py
```

```python
import torch
from glance_etd import EvidenceScorer, etd_select, strict_budget

tokens = torch.randn(1200, 2560)   # post-merger visual tokens, raster order (30 x 40)
question = torch.randn(12, 2560)   # input embeddings of the question
scorer = EvidenceScorer().eval()   # or EvidenceScorer.load("selector_epoch3.pt")
rows, cols = torch.arange(1200) // 40, torch.arange(1200) % 40
relevance, _ = scorer(tokens, question, rows, cols, 30, 40)
keep = etd_select(tokens, relevance, strict_budget(1200))   # 60 sorted indices
```

`tokens[keep]` is what the language model should see, together with the original positions of those indices.

## Reproducing the experiments

Training used 8×A800 and took about 4.7 hours for 175 steps. Evaluation used 4×A800. The stack is [Qwen3.5-4B](https://qwen.ai/blog?id=qwen3.5), `transformers` with Qwen3.5 support, `flash-attn`, and [verl](https://github.com/volcengine/verl)/FSDP with Ray.

**1. Code base.** Apply [`training/ltopd_glance_etd.patch`](training/ltopd_glance_etd.patch) from the training source root, the directory that contains `learnable_merge/`, `lt_opd/`, and `evaluation/`:

```bash
cd LT-OPD/src && patch -p1 < /path/to/Glance-ETD/training/ltopd_glance_etd.patch
```

The patch also keeps training-side options used by the supplementary ablations (`importance.*`, `opd_variant`, and the window encoder). They are off by default. Each recipe sets the options it uses.

<a id="2-data"></a>

**2. Data.** [`training/build_spatial_train.py`](training/build_spatial_train.py) builds the training mixture: 12.5K rows from the base set plus 1.5K spatial multiple-choice rows from the SAT static split and COCO train2017. Images used by GQA test-dev, POPE, MME, and CV-Bench are excluded. The script writes `DATA/data/LT-OPD-14K-spatial`.

```bash
LTOPD_DATA_ROOT=DATA LTOPD_SRC=LT-OPD/src python training/build_spatial_train.py
```

**3. MLLM.** Train the final recipe, then export the step-175 checkpoint:

```bash
cd LT-OPD/src
python -m lt_opd --implementation current train --model Qwen3.5-4B \
    --data-dir DATA/data/LT-OPD-14K-spatial \
    --output runs/glance_etd --config /path/to/training/configs/glance_etd.yaml
python -m lt_opd --implementation current export \
    --checkpoint runs/glance_etd/global_step_175 \
    --base-model Qwen3.5-4B --output exports/glance_etd
```

[`glance_etd_seed1.yaml`](training/configs/glance_etd_seed1.yaml) and [`glance_etd_seed2.yaml`](training/configs/glance_etd_seed2.yaml) repeat the run. [`dense_lr10x.yaml`](training/configs/dense_lr10x.yaml) is the dense-encoder variant at the same vision learning rate. [`dense_lr1x.yaml`](training/configs/dense_lr1x.yaml) and [`glance_lr1x.yaml`](training/configs/glance_lr1x.yaml) are the component ablations at the base vision learning rate. [`glance_trained_with_etd.yaml`](training/configs/glance_trained_with_etd.yaml) trains the glance model with ETD instead of DivPrune.

**4. Scorer.** One GPU, frozen Qwen3.5-4B. Train it on the untrained model's features and reuse that checkpoint for every Qwen model. `--no-question` trains the question-agnostic scorer reported in the supplementary.

```bash
python scorer/build_boxes.py TextVQA_0.5.1_train.json TextOCR_0.1_train.json \
    DATA/data/benchmarks/textocr/textvqa_train_boxes.json
python scorer/build_boxes.py TextVQA_0.5.1_val.json TextOCR_0.1_val.json \
    DATA/data/benchmarks/textocr/textvqa_val_boxes.json
python scorer/build_selector_data.py DATA
python scorer/train_selector.py scorer_out --root DATA --model Qwen3.5-4B \
    --data DATA/data/selector_train --epochs 3 --samples-per-epoch 60000 --general-weight 0.3
```

**5. Inference.** One shard, batch size 1, then score the predictions:

```bash
export LTOPD_SRC=LT-OPD/src GLANCE_ETD_SCORER=scorer_out/selector_epoch3.pt
python inference/glance_etd_runtime.py --dataset textvqa \
    --export-dir exports/glance_etd \
    --data-config DATA/data/benchmarks/datasets.json \
    --output-dir results/glance_etd/textvqa --rank 0 --world-size 1
python -m lt_opd score --dataset textvqa \
    --predictions results/glance_etd/textvqa --sources DATA/data/scorers \
    --output results/glance_etd/textvqa-score.json
```

| Variable | Default | Meaning |
|---|---|---|
| `GLANCE_ETD_BUDGET` | `ratio` | `ratio`: `K = ⌈rN⌉`. `floor`: `max(32, ⌈rN⌉)`. `power`: `K ∝ N^0.45`. |
| `GLANCE_ETD_RATIO` | `0.05` | Retention ratio `r`. |
| `GLANCE_ETD_SCALE` | unset | Scale of the power rule. Compute it with [`inference/power_scale.py`](inference/power_scale.py); it does not look at labels. |
| `GLANCE_ETD_GAMMA` | `0.25` | ETD exponent γ. |
| `GLANCE_ETD_STAGE1` | `scorer` | `diversity` uses the training-time cell rule. |
| `GLANCE_ETD_SELECT` | `etd` | `divprune` keeps the same per-image `K` with DivPrune. |

On a dense-encoder model the Stage-1 switch has no effect. Every selector compared in the paper keeps the same `K` per image.

## Related code and papers

Training uses [verl](https://github.com/volcengine/verl). Papers cited in the figures:

| Paper | Link |
|---|---|
| DivPrune, CVPR 2025 | [arXiv:2503.02175](https://arxiv.org/abs/2503.02175) |
| FastV, ECCV 2024 | [arXiv:2403.06764](https://arxiv.org/abs/2403.06764) |
| VisionZip, CVPR 2025 | [arXiv:2412.04467](https://arxiv.org/abs/2412.04467) |
| SparseVLM, ICML 2025 | [arXiv:2410.04417](https://arxiv.org/abs/2410.04417) |
| PyramidDrop, CVPR 2025 | [arXiv:2410.17247](https://arxiv.org/abs/2410.17247) |
| V\*, CVPR 2024 | [arXiv:2312.14135](https://arxiv.org/abs/2312.14135) |
| Qwen3.5 | [blog](https://qwen.ai/blog?id=qwen3.5) |
| Gemma 4 | [arXiv:2607.02770](https://arxiv.org/abs/2607.02770) |

## Citation

```bibtex
@inproceedings{wang2026glanceetd,
  title     = {Glance, then Select: Evidence-Tempered Visual Token Selection for Extremely Compressed Multimodal LLMs},
  author    = {Wang, Yuxuan and Zhao, Guangxiang and Tian, Yuxuan and Han, Yahui and Tan, Xinyu and Wang, Feiyu and Lv, Zongwei and Zhang, Xiangzheng and Sun, Lin and Yang, Tong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2027}
}
```

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE). The training patch does not relicense upstream code, third-party datasets, or model weights.

