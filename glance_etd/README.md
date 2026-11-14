# glance_etd

Self-contained PyTorch reference implementation of Glance-ETD (only `torch` is required).

- `etd.py`: DivPrune max–min greedy, the ETD weights `(N p_i)^γ`, ETD selection, and the budget rules.
- `scorer.py`: the evidence scorer `f_φ`, box-coverage targets and the listwise loss.
- `glance.py`: the two-pass glance encoder for a Qwen3.5 vision tower, with the training (diversity) and inference (scorer) cell rules.
- `teacher_kl.py`: the teacher-KL cost of a kept subset.
