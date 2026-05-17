"""Adaptation package — LoRA fine-tuning and linear embedding probes.

Implements the downstream adaptation strategies from PRAGMA paper
Section 3.1:

    Section 3.1.1 — Linear embedding probe:
        A frozen PRAGMA encoder with a single linear classification head.
        Used for rapid evaluation of embedding quality across tasks.
        Trains in minutes on small labelled datasets.

    Section 3.1.2 — LoRA fine-tuning:
        Parameter-efficient fine-tuning via Low-Rank Adaptation (LoRA).
        Updates a small number of task-specific parameters while keeping
        the PRAGMA encoder mostly frozen.
        Uses the Hugging Face PEFT library for LoRA implementation.

Reference: Ostroukhov et al. (2026), Section 3.1
LoRA reference: Hu et al. (2022), arXiv:2106.09685
"""

from .lora import apply_lora_to_pragma, PRAGMALoRAConfig
from .probe import LinearProbe

__all__ = [
    "apply_lora_to_pragma",
    "PRAGMALoRAConfig",
    "LinearProbe",
]
