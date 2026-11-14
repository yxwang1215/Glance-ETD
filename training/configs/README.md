# training/configs

| Recipe | Model |
|---|---|
| `glance_etd.yaml`, `glance_etd_seed{1,2}.yaml` | Glance-ETD MLLM, three seeds |
| `dense_lr10x.yaml`, `dense_lr10x_seed{1,2}.yaml` | Dense-encoder variant (same 10× vision learning rate) |
| `dense_lr1x.yaml`, `glance_lr1x.yaml` | Component ablation at the LT-OPD vision learning rate |
| `glance_trained_with_etd.yaml` | Glance MLLM trained with ETD instead of DivPrune (supplementary) |
