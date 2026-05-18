"""PRAGMA pretraining entrypoint.

Training script for the PRAGMA foundation model using the masked event
modelling (MEM) objective from Section 2.3.5.

This script provides a simplified training loop suitable for research
and smaller datasets. Full production training uses:
    - NeMo AutoModel for training orchestration
    - KFTO PyTorchJob for distributed training on OpenShift AI
    - KFP pipeline for orchestration (see pipeline/pragma_pipeline.py)

Prerequisites:
    1. Fit the tokeniser and build vocab.pkl:
           python src/data/fit_tokenizer.py
    2. Run training:
           python scripts/train_pragma.py \\
               --csv-path data/tabformer/card_transaction.v1.csv \\
               --vocab-path data/tabformer/vocab.pkl \\
               --output-dir outputs/pragma-s \\
               --epochs 10 \\
               --batch-size 32

Reference: Ostroukhov et al. (2026), Section 2.4 (Training Setup)
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PRAGMA, PRAGMAConfig
from src.model.assembler import EmbeddingAssembler
from src.masking import MaskingStrategy
from src.data import PragmaDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Number of events per sample — a 50-event window fits most customers and
# is well within PRAGMA-S's max_events=6500 capacity.
_NE_MAX = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRAGMA pretraining")
    parser.add_argument(
        "--csv-path", type=str, default="data/tabformer/card_transaction.v1.csv",
        help="Path to TabFormer card_transaction.v1.csv",
    )
    parser.add_argument(
        "--vocab-path", type=str, default="data/tabformer/vocab.pkl",
        help="Path to fitted pipeline vocab (produced by src/data/fit_tokenizer.py)",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/pragma-s",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (0 = main process only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'cpu', 'mps'")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log loss every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many global steps (0 = train all epochs)")
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
    logger.info(f"Training on device: {device}  num_workers={args.num_workers}")

    csv_path = Path(args.csv_path)
    vocab_path = Path(args.vocab_path)

    if not csv_path.exists():
        print(
            f"TabFormer CSV not found at {csv_path}\n"
            "Download from https://github.com/IBM/TabFormer/releases\n"
            f"and place it at that path before running."
        )
        sys.exit(1)

    if not vocab_path.exists():
        print(
            f"Vocab file not found at {vocab_path}\n"
            "Run: python src/data/fit_tokenizer.py"
        )
        sys.exit(1)

    # Datasets
    logger.info("Loading datasets ...")
    train_dataset = PragmaDataset(
        csv_path=csv_path,
        vocab_path=vocab_path,
        split="train",
        ne_max=_NE_MAX,
        ni_max=PRAGMAConfig.pragma_s().max_event_tokens,
    )
    val_dataset = PragmaDataset(
        csv_path=csv_path,
        vocab_path=vocab_path,
        split="val",
        ne_max=_NE_MAX,
        ni_max=PRAGMAConfig.pragma_s().max_event_tokens,
    )
    logger.info(f"Train customers: {len(train_dataset)}, val customers: {len(val_dataset)}")

    if len(train_dataset) == 0:
        logger.error(
            "train_dataset is empty — nothing to train on. "
            "Check --csv-path and --vocab-path."
        )
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Model config — PRAGMA-S
    config = PRAGMAConfig.pragma_s()
    ni_max = config.max_event_tokens

    # Vocabulary spec from the fitted pipeline
    import pickle
    with open(vocab_path, "rb") as f:
        pipeline = pickle.load(f)
    vocab_spec = pipeline.vocabulary_spec()

    # Assembler — owns the shared embedding table E; must be optimised alongside model
    assembler = EmbeddingAssembler(vocab_spec, config).to(device)
    assembler.train()

    # Model
    model = PRAGMA(config).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"PRAGMA-S: {n_params:,} parameters")

    # Masker
    masker = MaskingStrategy(config)

    # Optimiser — includes both model and assembler (assembler has the embedding table E)
    optimizer = AdamW(
        list(model.parameters()) + list(assembler.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training ...")
    global_step = 0
    _first_batch = True

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Move batch to device
            xe_val_ids = batch["xe_val_ids"].to(device)   # (B, ne_max, ni_max)
            xe_key_ids = batch["xe_key_ids"].to(device)
            xe_pos_ids = batch["xe_pos_ids"].to(device)
            xe_valid   = batch["xe_valid"].to(device)     # (B, ne_max, ni_max) bool
            xt         = batch["xt"].to(device)           # (B, ne_max, 3)
            te         = batch["te"].to(device)           # (B, 1+ne_max)
            xa_key_ids = batch["xa_key_ids"].to(device)   # (B, 1)
            xa_val_ids = batch["xa_val_ids"].to(device)
            xa_pos_ids = batch["xa_pos_ids"].to(device)
            ta         = batch["ta"].to(device)           # (B, 1)

            # Apply three-strategy masked event modelling
            masked_val_ids, _target_ids, mlm_mask = masker.forward(
                xe_val_ids, xe_key_ids
            )

            # Zero out masking for padding token positions (xe_valid=False)
            # so padded slots never contribute to the MLM loss
            mlm_mask = mlm_mask & xe_valid

            if not mlm_mask.any():
                if _first_batch:
                    logger.warning(
                        "First batch: all MLM-mask positions are empty after "
                        "masking + xe_valid filter — check masking strategy config."
                    )
                _first_batch = False
                continue
            _first_batch = False

            # Assemble embeddings — uses xe_val_ids (pre-mask) as targets
            assembled = assembler.forward(
                xa_key_ids=xa_key_ids,
                xa_val_ids=xa_val_ids,
                xa_pos_ids=xa_pos_ids,
                ta=ta,
                xe_key_ids=xe_key_ids,
                xe_val_ids=masked_val_ids,
                xe_pos_ids=xe_pos_ids,
                xt=xt,
                te=te,
                target_ids=xe_val_ids,
                mask=mlm_mask,
            )

            # Forward pass through PRAGMA
            # xe_valid.any(dim=-1): (B, ne) bool — True if event has ≥1 real token
            output = model.forward(
                xa=assembled.xa,
                ta=assembled.ta,
                xe=assembled.xe,
                xt=assembled.xt,
                te=assembled.te,
                mask=assembled.mlm_mask,
                event_valid=xe_valid.any(dim=-1),
            )

            if "logits" not in output:
                continue

            # MLM loss — only at masked positions
            valid_targets = assembled.targets[assembled.mlm_mask]
            loss = model.mlm_head.compute_loss(output["logits"], valid_targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(assembler.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            if args.max_steps > 0 and global_step >= args.max_steps:
                logger.info(f"Reached --max-steps {args.max_steps}; stopping early.")
                break

            if global_step % args.log_every == 0:
                logger.info(
                    f"epoch={epoch + 1}/{args.epochs}  "
                    f"step={global_step}  loss={loss_val:.4f}"
                )

        scheduler.step()

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} complete — "
                f"avg_loss={avg_loss:.4f}  steps={epoch_steps}"
            )

        # Checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "assembler_state_dict": assembler.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "vocab_spec": vocab_spec,
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved → {ckpt_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
