"""PRAGMA pretraining entrypoint.

Training script for the PRAGMA foundation model using the masked event
modelling (MEM) objective from Section 2.3.5.

Supports both single-process and multi-process distributed training (DDP).
When launched via torchrun (as done by the KFTO PyTorchJob), the script
detects RANK / LOCAL_RANK / WORLD_SIZE env vars and initialises DDP
automatically. No code changes are required between local and cluster runs.

Distributed behaviour:
  - All ranks load the same data; DistributedSampler assigns disjoint shards
  - model and assembler are wrapped in DistributedDataParallel
  - Logging, checkpointing, and S3 uploads are gated to rank 0 only
  - Checkpoints are uploaded to S3 after each local save when
    --s3-checkpoint-prefix is set (enables pod-restart recovery)
  - On startup with --resume, the script downloads the latest checkpoint
    from S3 (if --s3-checkpoint-prefix is set) before training begins

Single-process usage:
    python scripts/train_pragma.py \
        --csv-path data/tabformer/card_transaction.v1.csv \
        --vocab-path data/tabformer/vocab.pkl \
        --output-dir outputs/pragma-s \
        --epochs 10 \
        --batch-size 32

Multi-node usage (via torchrun — normally invoked by KFTO):
    torchrun \
        --nproc_per_node=4 --nnodes=4 \
        --node_rank=${RANK} \
        --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
        scripts/train_pragma.py \
        --model-variant pragma-m \
        --csv-path  /workspace/data/tabformer/card_transaction.v1.csv \
        --vocab-path /workspace/data/tabformer/vocab.pkl \
        --output-dir /workspace/outputs/pragma-m \
        --s3-checkpoint-prefix pragma-encoder/checkpoints/pragma-m \
        --resume \
        --epochs 10 --batch-size 32

Prerequisites:
  1. Fit the tokeniser and build vocab.pkl:
         python src/data/fit_tokenizer.py
  2. Upload data to S3 (for cluster jobs):
         python scripts/upload_training_data.py

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    return module.module if isinstance(module, DDP) else module


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_client():
    """Build boto3 client from MODEL_REGISTRY_* env vars.

    Returns (client, bucket) or (None, None) if credentials are absent.
    """
    bucket   = os.environ.get("MODEL_REGISTRY_BUCKET")
    endpoint = os.environ.get("MODEL_REGISTRY_ENDPOINT")
    access   = os.environ.get("MODEL_REGISTRY_ACCESS_KEY")
    secret   = os.environ.get("MODEL_REGISTRY_SECRET_KEY")
    if not all([bucket, endpoint, access, secret]):
        return None, None
    try:
        import boto3  # type: ignore[import]
    except ImportError:
        logger.warning("boto3 not installed — S3 checkpoint operations disabled.")
        return None, None
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{endpoint}",
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="us-west-2",
    )
    return client, bucket


def _upload_checkpoint(ckpt_path: Path, s3_prefix: str) -> None:
    """Upload a local checkpoint file to S3. Rank-0 only."""
    client, bucket = _s3_client()
    if client is None:
        logger.warning("S3 credentials not configured; skipping checkpoint upload.")
        return
    s3_key = f"{s3_prefix}/{ckpt_path.name}"
    logger.info(f"Uploading checkpoint -> s3://{bucket}/{s3_key}")
    client.upload_file(str(ckpt_path), bucket, s3_key)
    logger.info("Checkpoint uploaded.")


def _download_latest_checkpoint(output_dir: Path, s3_prefix: str) -> Optional[Path]:
    """Download the most recent checkpoint from S3 to output_dir.

    Returns the local path of the downloaded file, or None if no checkpoint
    exists in S3 or credentials are absent.
    """
    client, bucket = _s3_client()
    if client is None:
        return None
    response = client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix + "/")
    objects = [o for o in response.get("Contents", []) if o["Key"].endswith(".pt")]
    if not objects:
        logger.info(f"No checkpoints found in s3://{bucket}/{s3_prefix}/")
        return None
    latest = max(objects, key=lambda o: o["LastModified"])
    fname = Path(latest["Key"]).name
    local = output_dir / fname
    logger.info(f"Downloading checkpoint from s3://{bucket}/{latest['Key']} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, latest["Key"], str(local))
    logger.info(f"Downloaded to {local}")
    return local


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

def parse_args() -> argparse.Namespace:
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
             "Credentials are read from MODEL_REGISTRY_* env vars.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint. Checks --output-dir first, "
             "then downloads from --s3-checkpoint-prefix if no local checkpoint "
             "is found and S3 credentials are configured.",
    )
    return parser.parse_args()


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
# Checkpoint resume — find latest local checkpoint
# ---------------------------------------------------------------------------

def _find_latest_local_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = sorted(output_dir.glob("checkpoint_epoch*.pt"))
    return checkpoints[-1] if checkpoints else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

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
                "Run: python src/data/fit_tokenizer.py"
            )
            sys.exit(1)
        output_dir.mkdir(parents=True, exist_ok=True)

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
        ckpt = _find_latest_local_checkpoint(output_dir)
        if ckpt is None and args.s3_checkpoint_prefix and is_rank0:
            ckpt = _download_latest_checkpoint(output_dir, args.s3_checkpoint_prefix)
        if distributed:
            # Broadcast whether rank 0 found a checkpoint
            found = torch.tensor(1 if ckpt is not None else 0, device=device)
            dist.broadcast(found, src=0)
            if not is_rank0 and found.item() == 1:
                # Non-rank-0 workers: wait for rank 0 to download, then find locally
                if distributed:
                    dist.barrier()
                ckpt = _find_latest_local_checkpoint(output_dir)
            elif is_rank0 and found.item() == 1:
                dist.barrier()
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

            if args.s3_checkpoint_prefix:
                _upload_checkpoint(ckpt_path, args.s3_checkpoint_prefix)

        # All ranks sync after checkpoint before next epoch
        if distributed:
            dist.barrier()

    if is_rank0:
        logger.info("Training complete.")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
