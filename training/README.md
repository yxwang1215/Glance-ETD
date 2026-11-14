# training

- `ltopd_glance_etd.patch`: patch to the public LT-OPD code release, applied from its source root. It adds the glance encoder on the student path (`efficient_vit.py`), the LT-OPD-14K-spatial mixture, and default-off options used by the supplementary ablations.
- `configs/`: training recipes, applied as overlays on `learnable_merge/training/default.yaml`.
- `build_spatial_train.py`: builds the LT-OPD-14K-spatial training set.
