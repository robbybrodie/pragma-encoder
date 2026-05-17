"""LoRA fine-tuning for PRAGMA.

Implements parameter-efficient fine-tuning of the PRAGMA model via
Low-Rank Adaptation (LoRA), as described in PRAGMA paper Section 3.1.2.

LoRA inserts trainable low-rank matrices into the attention layers
of the History Encoder (and optionally the Event Encoder). The base
PRAGMA weights remain frozen during fine-tuning, with only the LoRA
adapters and the task-specific head being trained.

This implementation uses the Hugging Face PEFT library for LoRA,
which provides production-quality LoRA with support for:
    - Rank decomposition in Q, K, V, and output projections
    - LoRA dropout for regularisation
    - Adapter merging for inference efficiency

Fine-tuning targets (from Section 3.1.2):
    - History Encoder attention projections (primary target)
    - Event Encoder attention projections (optional, for full adaptation)
    - Profile State Encoder is typically kept frozen

Reference: Ostroukhov et al. (2026), Section 3.1.2
LoRA reference: Hu et al. (2022), arXiv:2106.09685
PEFT library: https://github.com/huggingface/peft
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch.nn as nn


@dataclass
class PRAGMALoRAConfig:
    """Configuration for LoRA fine-tuning of PRAGMA.

    Attributes:
        r: LoRA rank. Typical values: 8, 16, 32.
           Higher rank = more expressive but more parameters.
        lora_alpha: LoRA scaling factor. Effective LR scale = alpha/r.
        lora_dropout: Dropout applied to LoRA activations. Default: 0.1.
        target_modules: Which attention projection modules to adapt.
            Defaults to History Encoder query and value projections.
        bias: Whether to train bias parameters. Options: 'none', 'all',
              'lora_only'. Default: 'none'.
        n_classes: Number of output classes for the task head.
        task_type: Task type ('binary', 'multiclass', 'regression').
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: [
            "self_attn.in_proj_weight",  # History Encoder self-attention
            "cross_attn.in_proj_weight", # History Encoder cross-attention
        ]
    )
    bias: str = "none"
    n_classes: int = 2
    task_type: str = "binary"


def apply_lora_to_pragma(
    model: nn.Module,
    config: PRAGMALoRAConfig,
) -> nn.Module:
    """Wrap a PRAGMA model with LoRA adapters using Hugging Face PEFT.

    Freezes all base model parameters and inserts trainable LoRA
    adapter matrices into the specified target modules.

    Args:
        model: A PRAGMA model instance (fully initialised, optionally
               loaded from a pretrained checkpoint).
        config: PRAGMALoRAConfig specifying LoRA hyperparameters.

    Returns:
        The PRAGMA model wrapped with PEFT LoRA adapters.
        Only LoRA parameters and the task head are trainable.

    Raises:
        ImportError: If the `peft` library is not installed.
            Install with: pip install peft>=0.10.0
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "The `peft` library is required for LoRA fine-tuning. "
            "Install it with: pip install peft>=0.10.0"
        ) from e

    # Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        # PRAGMA is an encoder model — use FEATURE_EXTRACTION task type
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a (possibly LoRA-wrapped) model.

    Args:
        model: Any nn.Module.

    Returns:
        Number of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
