"""PRAGMA pretraining entrypoint.

Training script for the PRAGMA foundation model using the masked event
modelling (MEM) objective from Section 2.3.5.

This script provides a simplified training loop suitable for research
and smaller datasets. Full production training uses:
    - NeMo AutoModel for training orchestration
    - KFTO PyTorchJob for distributed training on OpenShift AI
    - KFP pipeline for orchestration (see pipeline/pragma_pipeline.py)

Usage:
    python scripts/train_pragma.py \\
        --config configs/pragma_s.yaml \\
        --data-dir /data/transactions \\
        --output-dir /outputs/pragma-s \\
        --epochs 10 \\
        --batch-size 32

Reference: Ostroukhov et al. (2026), Section 2.4 (Training Setup)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PRAGMA, PRAGMAConfig
from src.masking import MaskingStrategy
from src.training import MaskedEventModellingLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRAGMA pretraining")
    parser.add_argument("--config", type=str, default="configs/pragma_s.yaml",
                        help="Model config YAML (pragma_s/m/l)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing tokenised transaction data")
    parser.add_argument("--output-dir", type=str, default="outputs/pragma-s",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'cpu', 'mps'")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    logger.info(f"Training on device: {device}")

    # Load config — use PRAGMA-S for now
    # TODO: parse YAML config file
    config = PRAGMAConfig.pragma_s()

    # Build model
    model = PRAGMA(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"PRAGMA model: {n_params:,} parameters")

    # Training objective
    criterion = MaskedEventModellingLoss(
        vocab_size=config.value_vocab_size,
    )

    # Masker — MaskingStrategy(config) uses config.token_mask_prob internally.
    # Call masker.forward(token_ids, key_ids) to obtain masked inputs and targets.
    masker = MaskingStrategy(config)

    # Optimiser
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training loop placeholder — implement DataLoader integration.")
    logger.info(f"Config: {config}")
    logger.info(f"Output: {output_dir}")

    # TODO: Implement DataLoader, training loop, evaluation, and checkpointing.
    # See docs/training-guide.md for the intended training loop structure.


if __name__ == "__main__":
    main()
