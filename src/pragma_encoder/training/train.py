"""PRAGMA pretraining entrypoint — installed as the ``pragma-encoder-train`` console script.

Training module for the PRAGMA foundation model using the masked event
modelling (MEM) objective from Section 2.3.5.

Supports both single-process and multi-process distributed training (DDP).
When launched via torchrun (as done by the KFTO PyTorchJob), the module
detects RANK / LOCAL_RANK / WORLD_SIZE env vars and initialises DDP
automatically. No code changes are required between local and cluster runs.

Distributed behaviour:
  - All ranks load the same data; DistributedSampler assigns disjoint shards
  - model and assembler are wrapped in DistributedDataParallel
  - Logging, checkpointing, and S3 uploads are gated to rank 0 only
  - Checkpoints are uploaded to S3 after each local save when
    --s3-checkpoint-prefix is set (enables pod-restart recovery)
  - On startup with --resume, the module downloads the latest checkpoint
    from S3 (if --s3-checkpoint-prefix is set) before training begins

Metrics output:
  - <output_dir>/metrics.jsonl is written by rank 0 at every training step
    and at every epoch boundary. Each line is a JSON object with keys:
      step, epoch, train_loss, learning_rate, timestamp
    Checkpoint events include a non-null checkpoint_path field.
  - <output_dir>/metadata.json is written at training completion.
  - Use ``python -m pragma_encoder.evaluation.plot_loss`` to generate
    a loss curve PNG from metrics.jsonl.

Single-process usage:
    pragma-encoder-train \\
        --csv-path data/tabformer/card_transaction.v1.csv \\
        --vocab-path data/tabformer/vocab.pkl \\
        --output-dir outputs/pragma-s \\
        --epochs 10 \\
        --batch-size 32

Multi-node usage (via torchrun — normally invoked by KFTO):
    torchrun \\
        --nproc_per_node=4 --nnodes=4 \\
        --node_rank=${RANK} \\
        --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \\
        -m pragma_encoder.training.train \\
        --model-variant pragma-m \\
        --csv-path  /workspace/data/tabformer/card_transaction.v1.csv \\
        --vocab-path /workspace/data/tabformer/vocab.pkl \\
        --output-dir /workspace/outputs/pragma-m \\
        --s3-checkpoint-prefix pragma-encoder/checkpoints/pragma-m \\
        --resume \\
        --epochs 10 --batch-size 32

Prerequisites:
  1. Fit the tokeniser and build vocab.pkl:
         python -m pragma_encoder.data.fit_tokenizer
  2. Upload data to S3 (for cluster jobs):
         python scripts/upload_training_data.py

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pragma_encoder.data import PragmaDataset
from pragma_encoder.masking import MaskingStrategy
from pragma_encoder.model import PRAGMA, PRAGMAConfig
from pragma_encoder.model.assembler import EmbeddingAssembler
from pragma_encoder.training.checkpoints import (
    resolve_resume_checkpoint,
    upload_checkpoint_if_rank0,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Number of events per sample — a 50-event window fits most customers and
# is well within PRAGMA-S's max_events=6500 capacity.
_NE_MAX = 50


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _append_metrics(path: "Path | None", record: dict) -> None:
    """Append one record to metrics.jsonl (rank-0 only; no-op if path is None).

    Each call opens, appends, and closes the file so the JSONL is always
    flushed to disk — callers do not need to manage a file handle.
    """
    if path is None:
        return
    with open(path, "a") as _f:
        _f.write(json.dumps(record) + "\n")


_CONFIGS = {
    "pragma-s": PRAGMAConfig.pragma_s,
    "pragma-m": PRAGMAConfig.pragma_m,
    "pragma-l": PRAGMAConfig.pragma_l,
}


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _is_distributed() -> bool:
    """True when launched by torchrun with WORLD_SIZE > 1."""
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def _init_distributed() -> tuple[int, int, int]:
    """Initialise distributed process group. Returns (rank, local_rank, world_size).

    Uses nccl backend when CUDA GPUs are available (production), gloo otherwise
    (CPU-only smoke tests and development runs without GPUs).
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when wrapped in DDP, else the module itself."""
    return module.module if isinstance(module, DDP) else module  # type: ignore[return-value]


def _load_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    assembler: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    """Load checkpoint into model, assembler, and optimizer.

    Returns (start_epoch, global_step).
    """
    logger.info(f"Resuming from checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    _unwrap(model).load_state_dict(state["model_state_dict"])
    _unwrap(assembler).load_state_dict(state["assembler_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    start_epoch = state["epoch"]        # epoch after which this was saved
    global_step = state["global_step"]
    logger.info(f"Resumed from epoch {start_epoch}, global_step {global_step}")
    return start_epoch, global_step


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRAGMA pretraining")
    parser.add_argument(
        "--model-variant",
        type=str,
        default="pragma-s",
        choices=list(_CONFIGS),
        help="Model scale: pragma-s (~10M), pragma-m (~100M), pragma-l (~1B)",
    )
    parser.add_argument(
        "--csv-path", type=str, default="data/tabformer/card_transaction.v1.csv",
        help="Path to TabFormer card_transaction.v1.csv",
    )
    parser.add_argument(
        "--vocab-path", type=str, default="data/tabformer/vocab.pkl",
        help=(
            "Path to fitted pipeline vocab; "
            "produce with: python -m pragma_encoder.data.fit_tokenizer"
        ),
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
                        help="Device: 'auto', 'cuda', 'cpu', 'mps'. "
                             "Ignored in distributed mode (LOCAL_RANK is used).")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log loss every N steps (rank 0 only)")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Save checkpoint every N epochs (rank 0 only)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many global steps (0 = train all epochs)")
    parser.add_argument(
        "--s3-checkpoint-prefix",
        type=str,
        default="",
        help="S3 key prefix for checkpoint uploads, e.g. "
             "'pragma-encoder/checkpoints/pragma-s'. "
             "If set, each checkpoint is uploaded to S3 immediately after saving. "
             "Credentials are read from AWS_* env vars (native RHOAI S3 Connection schema).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint. Checks --output-dir first, "
             "then downloads from --s3-checkpoint-prefix if no local checkpoint "
             "is found and S3 credentials are configured.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Cap the training dataset to this many customers (0 = use all). "
             "Use for smoke/tiny runs that must produce a real artifact quickly "
             "without the full dataset. Validation set size is not affected.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Dataset reference to record in metadata.json (e.g. S3 URI or dataset name). "
             "Purely for metadata — does not affect training logic. "
             "Set by the pipeline component to record the original dataset source; "
             "leave empty for local runs (csv-path is recorded instead).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(device_arg: str, local_rank: int, distributed: bool) -> torch.device:
    if distributed:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

# Keys whose values must never appear in metadata.json.
# Training CLI args do not currently include credentials (AWS_* are env vars,
# not CLI args), but this explicit allowlist documents the contract and guards
# against future regressions if credential-adjacent args are added.
_METADATA_ALLOWED_ARG_KEYS: frozenset[str] = frozenset({
    "model_variant",
    "epochs",
    "batch_size",
    "lr",
    "max_steps",
    "limit_rows",
    "seed",
    "num_workers",
    "checkpoint_every",
    "s3_checkpoint_prefix",   # S3 prefix path — not a credential
    "dataset_name",            # dataset reference label — not a credential
    # Deliberately excluded: csv_path, vocab_path, output_dir, device, resume
    # (staging paths already recorded in dataset/output_dir sections)
})


def _safe_args_for_metadata(args: argparse.Namespace) -> dict:
    """Return a sanitised subset of training args safe to record in metadata.json.

    Only keys in ``_METADATA_ALLOWED_ARG_KEYS`` are included.  Any future arg
    whose value could contain a credential (access key, secret, token, password)
    is absent from the allowlist and therefore absent from metadata.

    This is not a security boundary — credentials should never be CLI args.
    It is a defence-in-depth documentation contract.
    """
    return {
        k: getattr(args, k)
        for k in _METADATA_ALLOWED_ARG_KEYS
        if hasattr(args, k)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- Distributed setup ------------------------------------------------
    distributed = _is_distributed()
    if distributed:
        rank, local_rank, world_size = _init_distributed()
    else:
        rank, local_rank, world_size = 0, 0, 1

    is_rank0 = (rank == 0)

    torch.manual_seed(args.seed + rank)  # different seed per rank for dropout
    device = get_device(args.device, local_rank, distributed)

    if is_rank0:
        logger.info(
            f"Training on device: {device}  "
            f"distributed={distributed}  world_size={world_size}  "
            f"num_workers={args.num_workers}"
        )

    # ---- Paths ------------------------------------------------------------
    csv_path   = Path(args.csv_path)
    vocab_path = Path(args.vocab_path)
    output_dir = Path(args.output_dir)

    if is_rank0:
        if not csv_path.exists():
            print(
                f"TabFormer CSV not found at {csv_path}\n"
                "Download from https://github.com/IBM/TabFormer/releases\n"
                "or run: python scripts/upload_training_data.py"
            )
            sys.exit(1)
        if not vocab_path.exists():
            print(
                f"Vocab file not found at {vocab_path}\n"
                "Run: python -m pragma_encoder.data.fit_tokenizer"
            )
            sys.exit(1)
        output_dir.mkdir(parents=True, exist_ok=True)

    # metrics.jsonl — rank 0 only; None on worker ranks (no-op in _append_metrics)
    metrics_file: Optional[Path] = (output_dir / "metrics.jsonl") if is_rank0 else None

    # All ranks wait until rank 0 has validated paths
    if distributed:
        dist.barrier()

    # ---- Model config -----------------------------------------------------
    config = _CONFIGS[args.model_variant]()
    if is_rank0:
        logger.info(f"Model variant: {args.model_variant}")

    # ---- Datasets ---------------------------------------------------------
    if is_rank0:
        logger.info("Loading datasets ...")

    train_dataset = PragmaDataset(
        csv_path=csv_path,
        vocab_path=vocab_path,
        split="train",
        ne_max=_NE_MAX,
        ni_max=config.max_event_tokens,
    )
    val_dataset = PragmaDataset(
        csv_path=csv_path,
        vocab_path=vocab_path,
        split="val",
        ne_max=_NE_MAX,
        ni_max=config.max_event_tokens,
    )

    # Apply --limit-rows: cap training dataset to N customers (0 = no cap)
    if args.limit_rows > 0 and len(train_dataset) > args.limit_rows:
        from torch.utils.data import Subset  # noqa: PLC0415
        train_dataset = Subset(train_dataset, list(range(args.limit_rows)))
        if is_rank0:
            logger.info(f"--limit-rows {args.limit_rows}: training on {args.limit_rows} customers")

    if is_rank0:
        logger.info(f"Train customers: {len(train_dataset)}, val: {len(val_dataset)}")
        if len(train_dataset) == 0:
            logger.error(
                "train_dataset is empty — nothing to train on. "
                "Check --csv-path and --vocab-path."
            )
            sys.exit(1)

    # ---- DataLoader -------------------------------------------------------
    if distributed:
        train_sampler: Optional[DistributedSampler] = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ---- Vocabulary spec --------------------------------------------------
    import pickle
    with open(vocab_path, "rb") as f:
        pipeline = pickle.load(f)
    vocab_spec = pipeline.vocabulary_spec()

    # ---- Model and assembler ----------------------------------------------
    assembler = EmbeddingAssembler(vocab_spec, config).to(device)
    assembler.train()

    model = PRAGMA(config).to(device)
    model.train()

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"{args.model_variant.upper()}: {n_params:,} parameters")

    # ---- Optimiser (before DDP, so optimizer sees original parameters) ----
    optimizer = AdamW(
        list(model.parameters()) + list(assembler.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Resume from checkpoint -------------------------------------------
    start_epoch = 0
    global_step = 0

    if args.resume:
        # TD-006 fix: all ranks independently download the checkpoint from S3.
        # Old pattern (broken): only rank 0 downloads; workers search their empty emptyDir.
        # New pattern: rank 0 selects key via S3, broadcasts key string to all ranks,
        # ALL ranks independently call download_checkpoint_for_rank, then barrier.
        ckpt = resolve_resume_checkpoint(
            output_dir=output_dir,
            s3_prefix=args.s3_checkpoint_prefix,
            rank=rank,
            distributed=distributed,
            device=device,
        )
        if ckpt is not None:
            start_epoch, global_step = _load_checkpoint(
                ckpt, model, assembler, optimizer, device
            )

    # ---- DDP wrap (after loading checkpoint, before training) -------------
    if distributed:
        _ddp_ids = [local_rank] if torch.cuda.is_available() else None
        model    = DDP(model,    device_ids=_ddp_ids)
        assembler = DDP(assembler, device_ids=_ddp_ids)

    # ---- Masker -----------------------------------------------------------
    masker = MaskingStrategy(config)

    # ---- Training loop ----------------------------------------------------
    if is_rank0:
        logger.info("Starting training ...")

    epoch = start_epoch - 1  # tracks last completed epoch index; -1 if none run
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # ensures different shuffle each epoch

        epoch_loss  = 0.0
        epoch_steps = 0
        _first_batch = True

        for batch in train_loader:
            optimizer.zero_grad()

            # Move batch to device
            xe_val_ids = batch["xe_val_ids"].to(device)
            xe_key_ids = batch["xe_key_ids"].to(device)
            xe_pos_ids = batch["xe_pos_ids"].to(device)
            xe_valid   = batch["xe_valid"].to(device)
            xt         = batch["xt"].to(device)
            te         = batch["te"].to(device)
            xa_key_ids = batch["xa_key_ids"].to(device)
            xa_val_ids = batch["xa_val_ids"].to(device)
            xa_pos_ids = batch["xa_pos_ids"].to(device)
            ta         = batch["ta"].to(device)

            # Three-strategy masked event modelling
            masked_val_ids, _target_ids, mlm_mask = masker.forward(
                xe_val_ids, xe_key_ids
            )
            # Zero masking at padding positions
            mlm_mask = mlm_mask & xe_valid

            if not mlm_mask.any():
                if _first_batch and is_rank0:
                    logger.warning(
                        "First batch: all MLM-mask positions are empty after "
                        "masking + xe_valid filter — check masking strategy config."
                    )
                _first_batch = False
                continue
            _first_batch = False

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

            valid_targets = assembled.targets[assembled.mlm_mask]
            loss = _unwrap(model).mlm_head.compute_loss(output["logits"], valid_targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(_unwrap(model).parameters()) + list(_unwrap(assembler).parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            loss_val     = loss.item()
            epoch_loss  += loss_val
            epoch_steps += 1
            global_step += 1

            # Write per-step metrics to metrics.jsonl (rank 0 only)
            if is_rank0:
                _current_lr = optimizer.param_groups[0]["lr"]
                _append_metrics(metrics_file, {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": round(loss_val, 6),
                    "learning_rate": _current_lr,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "checkpoint_path": None,
                })

            if args.max_steps > 0 and global_step >= args.max_steps:
                if is_rank0:
                    logger.info(f"Reached --max-steps {args.max_steps}; stopping early.")
                break

            if is_rank0 and global_step % args.log_every == 0:
                logger.info(
                    f"epoch={epoch + 1}/{args.epochs}  "
                    f"step={global_step}  loss={loss_val:.4f}"
                )

        scheduler.step()

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        if is_rank0 and epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} complete — "
                f"avg_loss={avg_loss:.4f}  steps={epoch_steps}"
            )
            # Write epoch-end summary to metrics.jsonl
            _append_metrics(metrics_file, {
                "step": global_step,
                "epoch": epoch + 1,
                "event": "epoch_end",
                "avg_train_loss": round(avg_loss, 6),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            })

        # ---- Checkpoint (rank 0 only) -------------------------------------
        if is_rank0 and (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict":     _unwrap(model).state_dict(),
                    "assembler_state_dict": _unwrap(assembler).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config":     config,
                    "vocab_spec": vocab_spec,
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved -> {ckpt_path}")
            # Record checkpoint event in metrics.jsonl
            _append_metrics(metrics_file, {
                "step": global_step,
                "epoch": epoch + 1,
                "event": "checkpoint",
                "checkpoint_path": str(ckpt_path),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            })

            if args.s3_checkpoint_prefix:
                upload_checkpoint_if_rank0(ckpt_path, args.s3_checkpoint_prefix, is_rank0=True)

        # All ranks sync after checkpoint before next epoch
        if distributed:
            dist.barrier()

    # ---- Write metadata.json (rank 0 only) -----------------------------------
    if is_rank0:
        ckpt_files = sorted(output_dir.glob("checkpoint_epoch*.pt"))
        final_checkpoint = str(ckpt_files[-1]) if ckpt_files else None

        # dataset section: record original reference (dataset_name) and staging path.
        # csv_staging_path is the local path used for this run — for local runs it is
        # the user-provided path; for pipeline runs it is the ephemeral scratch path.
        # dataset_name records the original source reference (S3 URI, dataset label)
        # set by the caller via --dataset-name; empty string for direct local runs.
        _dataset_ref = args.dataset_name if args.dataset_name else str(csv_path)
        _metrics_jsonl = str(output_dir / "metrics.jsonl")
        metadata: dict = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "model_variant": args.model_variant,
            "epochs_completed": epoch + 1,
            "global_step": global_step,
            "dataset": {
                "dataset_name": _dataset_ref,
                "csv_staging_path": str(csv_path),
                "limit_rows": args.limit_rows,
            },
            "output_dir": str(output_dir),
            "final_checkpoint": final_checkpoint,
            "metrics_jsonl": _metrics_jsonl,
            "args": _safe_args_for_metadata(args),
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata written -> {metadata_path}")
        if final_checkpoint:
            logger.info(f"Final checkpoint -> {final_checkpoint}")

    if is_rank0:
        logger.info("Training complete.")

    if distributed:
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
