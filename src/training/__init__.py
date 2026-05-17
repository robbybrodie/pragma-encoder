"""Training package — MLM objective, sequence packing, and batching.

Implements the training utilities described in PRAGMA paper Section 2.3.5
and Section 2.4 (training setup).

    objective.py — MLM loss with label smoothing (Section 2.3.5)
    packing.py   — Sequence packing for training efficiency (Section 2.4)
    batching.py  — Dynamic batching by sequence length (Section 2.4)

Training scale note (from the paper):
    Revolut pretrained on 207B tokens with LMDB storage and custom
    sequence packing. This implementation provides a simplified training
    loop suitable for research and smaller datasets. The architecture
    is identical; the training infrastructure is simplified.

Reference: Ostroukhov et al. (2026), Sections 2.3.5 and 2.4
"""

from .objective import MaskedEventModellingLoss
from .packing import SequencePacker
from .batching import DynamicBatchSampler

__all__ = [
    "MaskedEventModellingLoss",
    "SequencePacker",
    "DynamicBatchSampler",
]
