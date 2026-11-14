"""Reference implementation of Glance-ETD (pure PyTorch, no dependency on the training framework).

etd.py         Stage 2 selection (DivPrune greedy, ETD weights (N p)^gamma, budget rules)
scorer.py      evidence scorer f_phi, box-coverage targets and listwise loss
glance.py      Stage 1 two-pass glance encoder for a Qwen3.5-VL vision tower
teacher_kl.py  teacher-KL diagnostic
"""
from .etd import (GAMMA, divprune_select, etd_select, evidence_weights, floor_budget, max_min_diversity,
                  power_budget, strict_budget)
from .scorer import EvidenceScorer, box_coverage, listwise_loss
