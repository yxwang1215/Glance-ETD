# assets

Figures from the paper, copied as PNG for the README. They are illustrations of the method and the reported experiments. The code license does not relicense the figures separately from the repository.

| File | Paper figure | What it shows |
|---|---|---|
| `paper.pdf` | Camera-ready PDF | The manuscript linked from the README. |
| `overview.png` | Method overview | Training by on-policy self-distillation, Stage 1 glance selection by the scorer, and Stage 2 ETD with γ = 0.25. |
| `motivation.png` | Motivation | Encoder time on HR-Bench 4K, one TextVQA example, and answer-token coverage. |
| `budget.png` | Budget sweep | Retention of Qwen3.5-4B from 5% to 50% of the visual tokens, averaged over nine benchmarks. |
| `efficiency.png` | Efficiency | Retention against HR-Bench 4K speed-up, peak memory, and prefill and end-to-end speed-up. |
| `backbones.png` | Cross-backbone | Strict-5% retention on Qwen3.5-4B, Gemma 4 E4B, GLM-4.6V-Flash, and InternVL3.5-8B. |
| `gemma.png` | Gemma budgets | ETD and DivPrune on Gemma 4 E4B-it from 5% to 25% of the tokens. |
| `efficiency-supp.png` | Supplementary efficiency | Task-matched speed, prefill FLOPs, stage breakdown, throughput, memory, and decode cache. |
