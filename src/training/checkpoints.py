"""S3 checkpoint upload/download utilities for all-rank distributed resume.

Implements the all-rank checkpoint download pattern that fixes TD-006:

  Old (broken) pattern — emptyDir per pod:
    Rank 0 downloads checkpoint from S3 to its own emptyDir.
    Workers broadcast-receive a 'found' boolean tensor.
    Workers then call _find_latest_local_checkpoint() on their own emptyDir.
    Workers find nothing — emptyDir is per-pod; rank 0's download is invisible.
    Workers start from global_step=0. Model states diverge silently.

  New (correct) pattern — all-rank download:
    Rank 0 selects the latest checkpoint key from S3.
    Rank 0 broadcasts the key string to all ranks via dist.broadcast_object_list.
    EVERY rank independently downloads the checkpoint from S3 to its own path.
    All ranks call dist.barrier() after download.
    All ranks load from their own local copy.

No shared filesystem (emptyDir) is assumed between ranks.
No kfp / kfp-kubernetes is imported — this is a training-image-only module.
boto3 is imported lazily inside functions so the module can be loaded in
environments where boto3 is not installed (e.g. local dev without S3 access).

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
TD-006: docs/tech-debt.md — multi-node checkpoint resume with per-pod emptyDir
"""

from __future__ import annotations

import os
import pathlib
from typing import Optional

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# S3 configuration
# ---------------------------------------------------------------------------


class _S3Config(dict):
    """S3 configuration dict that redacts the secret key in repr.

    Behaves exactly like a dict for item access and containment tests.
    __repr__ replaces the secret_key value with '***REDACTED***' so that
    logging or test assertion output never leaks the raw credential.
    """

    def __repr__(self) -> str:
        safe = {
            k: ("***REDACTED***" if k == "secret_key" else v)
            for k, v in self.items()
        }
        return f"_S3Config({safe!r})"


def parse_s3_config_from_env() -> Optional[_S3Config]:
    """Parse S3 connection config from MODEL_REGISTRY_* environment variables.

    Required vars: MODEL_REGISTRY_BUCKET, MODEL_REGISTRY_ENDPOINT.
    Optional vars: MODEL_REGISTRY_ACCESS_KEY, MODEL_REGISTRY_SECRET_KEY.

    Returns:
        _S3Config dict with keys: bucket, endpoint, access_key, secret_key.
        None if MODEL_REGISTRY_BUCKET or MODEL_REGISTRY_ENDPOINT is absent/empty.
    """
    bucket = os.environ.get("MODEL_REGISTRY_BUCKET", "").strip()
    endpoint = os.environ.get("MODEL_REGISTRY_ENDPOINT", "").strip()
    if not bucket or not endpoint:
        return None
    return _S3Config(
        bucket=bucket,
        endpoint=endpoint,
        access_key=os.environ.get("MODEL_REGISTRY_ACCESS_KEY", "").strip(),
        secret_key=os.environ.get("MODEL_REGISTRY_SECRET_KEY", "").strip(),
    )


def _make_s3_client(config: _S3Config):
    """Create a boto3 S3 client from a parsed S3 config dict.

    boto3 is imported here (not at module level) so this module can be
    imported in environments where boto3 is not installed.

    Args:
        config: _S3Config dict from parse_s3_config_from_env().

    Returns:
        boto3 S3 client configured with the given endpoint and credentials.
    """
    import boto3  # noqa: PLC0415

    endpoint = config["endpoint"]
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
    )


# ---------------------------------------------------------------------------
# S3 key listing and selection
# ---------------------------------------------------------------------------


def list_checkpoint_keys(client, bucket: str, prefix: str) -> list[str]:
    """List all .pt checkpoint keys under prefix/ in the S3 bucket.

    Args:
        client: boto3 S3 client.
        bucket: S3 bucket name.
        prefix: Key prefix (trailing slash added automatically).

    Returns:
        Sorted list of S3 key strings ending in .pt.
        Empty list when no .pt objects exist or Contents key is absent.
    """
    prefix_with_slash = prefix.rstrip("/") + "/"
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix_with_slash)
    contents = response.get("Contents", [])
    return sorted(obj["Key"] for obj in contents if obj["Key"].endswith(".pt"))


def select_latest_checkpoint_key(client, bucket: str, prefix: str) -> Optional[str]:
    """Return the S3 key of the latest checkpoint, selected by LastModified.

    Args:
        client: boto3 S3 client.
        bucket: S3 bucket name.
        prefix: Key prefix.

    Returns:
        S3 key string of the most recently modified .pt file.
        None if no .pt objects exist under the prefix.
    """
    prefix_with_slash = prefix.rstrip("/") + "/"
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix_with_slash)
    contents = response.get("Contents", [])
    pt_objects = [obj for obj in contents if obj["Key"].endswith(".pt")]
    if not pt_objects:
        return None
    return max(pt_objects, key=lambda obj: obj["LastModified"])["Key"]


# ---------------------------------------------------------------------------
# Local checkpoint discovery
# ---------------------------------------------------------------------------


def _find_latest_local_checkpoint(output_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Return the lexicographically latest .pt file in output_dir.

    Used in single-node mode when S3 is not configured.
    Checkpoint filenames are designed to sort chronologically
    (e.g. checkpoint_epoch0001.pt < checkpoint_epoch0002.pt).

    Args:
        output_dir: Directory to scan for .pt files.

    Returns:
        Path to the latest checkpoint, or None if none exist.
    """
    pt_files = sorted(output_dir.glob("*.pt"))
    return pt_files[-1] if pt_files else None


# ---------------------------------------------------------------------------
# Upload — rank 0 only
# ---------------------------------------------------------------------------


def upload_checkpoint_if_rank0(
    ckpt_path: pathlib.Path,
    s3_prefix: str,
    is_rank0: bool,
) -> None:
    """Upload a checkpoint file to S3 if this process is rank 0.

    Only rank 0 uploads to avoid write races in distributed training.
    This is a no-op when is_rank0=False or when S3 config is absent.

    The S3 key is constructed as: <s3_prefix>/<filename>.

    Args:
        ckpt_path: Local path to the .pt checkpoint file.
        s3_prefix: S3 key prefix (e.g. 'pragma-encoder/checkpoints/pragma-s').
        is_rank0:  True for the master rank; False for worker ranks.
    """
    if not is_rank0:
        return
    config = parse_s3_config_from_env()
    if config is None:
        return
    client = _make_s3_client(config)
    s3_key = f"{s3_prefix.rstrip('/')}/{ckpt_path.name}"
    client.upload_file(str(ckpt_path), config["bucket"], s3_key)


# ---------------------------------------------------------------------------
# Download — ALL ranks (core TD-006 fix)
# ---------------------------------------------------------------------------


def download_checkpoint_for_rank(
    key: Optional[str],
    output_dir: pathlib.Path,
) -> Optional[pathlib.Path]:
    """Download a checkpoint from S3 to the local output directory.

    Called independently by EVERY rank. Each rank downloads to its own
    emptyDir — no shared filesystem between pods is assumed.

    This is the core of the TD-006 fix: workers do not rely on rank 0's
    local files; they each independently fetch the checkpoint from S3.

    Args:
        key:        S3 key of the checkpoint to download, or None.
        output_dir: Local directory to download into (created if absent).

    Returns:
        Local Path to the downloaded file.
        None if key is None or S3 config is absent.
    """
    if key is None:
        return None
    config = parse_s3_config_from_env()
    if config is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = pathlib.Path(key).name
    local_path = output_dir / filename
    client = _make_s3_client(config)
    client.download_file(config["bucket"], key, str(local_path))
    return local_path


# ---------------------------------------------------------------------------
# Distributed barrier helper
# ---------------------------------------------------------------------------


def barrier_if_distributed(distributed: bool) -> None:
    """Call dist.barrier() only when the process group is active.

    Centralises the distributed=True guard so callers do not repeat it.

    Args:
        distributed: True if the DDP process group has been initialised.
    """
    if distributed:
        dist.barrier()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def resolve_resume_checkpoint(
    output_dir: pathlib.Path,
    s3_prefix: str,
    rank: int,
    distributed: bool,
    device: torch.device,
) -> Optional[pathlib.Path]:
    """Resolve the checkpoint to resume from, using all-rank S3 download.

    This is the replacement for the broken TD-006 pattern. The key difference:
      - Old: only rank 0 downloads; workers search their own (empty) emptyDir.
      - New: rank 0 selects key; ALL ranks independently download from S3.

    Single-node mode (distributed=False):
      No dist.* calls are made (process group may not be initialised).
      If S3 is configured and s3_prefix is set, downloads the latest key.
      Otherwise falls back to the latest local .pt file in output_dir.

    Distributed mode (distributed=True):
      1. Rank 0 selects the latest checkpoint key from S3.
      2. Rank 0 broadcasts the key string to all ranks via broadcast_object_list.
      3. ALL ranks independently call download_checkpoint_for_rank.
      4. All ranks call dist.barrier() before returning.
      5. Returns the local path (or None if no checkpoint found).

    Args:
        output_dir:  Local directory to download the checkpoint into.
        s3_prefix:   S3 key prefix for checkpoint listing and download.
        rank:        This process's rank (0 = master).
        distributed: True if the DDP process group is initialised.
        device:      Torch device (reserved for future tensor operations).

    Returns:
        Local Path to the checkpoint file, or None if none found.
    """
    is_rank0 = (rank == 0)

    if not distributed:
        # Single-node: no dist calls, direct S3 or local fallback
        config = parse_s3_config_from_env()
        if config and s3_prefix:
            client = _make_s3_client(config)
            key = select_latest_checkpoint_key(client, config["bucket"], s3_prefix)
            if key:
                return download_checkpoint_for_rank(key, output_dir)
        return _find_latest_local_checkpoint(output_dir)

    # Distributed: rank 0 selects; all ranks receive via broadcast; all download
    key_holder: list[Optional[str]] = [None]

    if is_rank0:
        config = parse_s3_config_from_env()
        if config and s3_prefix:
            client = _make_s3_client(config)
            key_holder[0] = select_latest_checkpoint_key(
                client, config["bucket"], s3_prefix
            )
        # If no S3 config or no prefix, key_holder[0] stays None.
        # All ranks will receive None, skip download, and return None.

    # Broadcast the selected key string from rank 0 to all ranks.
    # dist.broadcast_object_list works with Python objects (strings, None),
    # unlike dist.broadcast which requires tensors of a fixed dtype.
    dist.broadcast_object_list(key_holder, src=0)
    key = key_holder[0]

    # ALL ranks independently download the checkpoint to their own emptyDir.
    # Workers cannot read rank 0's emptyDir — they must download themselves.
    result = download_checkpoint_for_rank(key, output_dir)

    # Synchronise: every rank must finish downloading before any rank loads.
    barrier_if_distributed(distributed)

    return result
