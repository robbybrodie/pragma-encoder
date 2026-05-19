"""Upload PRAGMA training data prerequisites to S3.

Uploads the TabFormer CSV and the fitted tokeniser vocab to the S3 paths
expected by the PyTorchJob init container. Run this once before submitting
any training job. The operation is idempotent.

Usage:
    # Export credentials (same keys as pragma-workbench-env secret):
    export MODEL_REGISTRY_BUCKET=<bucket>
    export MODEL_REGISTRY_ENDPOINT=<host-without-scheme>
    export MODEL_REGISTRY_ACCESS_KEY=<access-key>
    export MODEL_REGISTRY_SECRET_KEY=<secret-key>

    python scripts/upload_training_data.py \
        --csv-path   data/tabformer/card_transaction.v1.csv \
        --vocab-path data/tabformer/vocab.pkl

    # Dry run (no transfer):
    python scripts/upload_training_data.py --dry-run

S3 destination paths (relative to bucket root):
    pragma-encoder/data/tabformer/card_transaction.v1.csv
    pragma-encoder/data/tabformer/vocab.pkl

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
"""

import argparse
import os
import sys
from pathlib import Path

# S3 prefix under which all PRAGMA training inputs are stored.
# Matches the path expected by the PyTorchJob init container.
_S3_PREFIX = "pragma-encoder/data/tabformer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload PRAGMA training data prerequisites to S3"
    )
    p.add_argument(
        "--csv-path",
        type=str,
        default="data/tabformer/card_transaction.v1.csv",
        help="Local path to card_transaction.v1.csv",
    )
    p.add_argument(
        "--vocab-path",
        type=str,
        default="data/tabformer/vocab.pkl",
        help="Local path to fitted tokeniser vocab (produced by src/data/fit_tokenizer.py)",
    )
    p.add_argument(
        "--region",
        type=str,
        default="us-west-2",
        help="AWS region for the S3 endpoint",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without transferring anything",
    )
    return p.parse_args()


def _require_env() -> tuple[str, str, str, str]:
    """Read and validate MODEL_REGISTRY_* env vars. Exits on missing vars."""
    required = (
        "MODEL_REGISTRY_BUCKET",
        "MODEL_REGISTRY_ENDPOINT",
        "MODEL_REGISTRY_ACCESS_KEY",
        "MODEL_REGISTRY_SECRET_KEY",
    )
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(
            "ERROR: the following environment variables are not set:\n"
            + "\n".join(f"  {v}" for v in missing)
            + "\n\nExport MODEL_REGISTRY_* vars (same keys as pragma-workbench-env secret).",
            file=sys.stderr,
        )
        sys.exit(1)
    return (
        os.environ["MODEL_REGISTRY_BUCKET"],
        os.environ["MODEL_REGISTRY_ENDPOINT"],
        os.environ["MODEL_REGISTRY_ACCESS_KEY"],
        os.environ["MODEL_REGISTRY_SECRET_KEY"],
    )


def _s3_client(endpoint: str, access: str, secret: str, region: str):
    try:
        import boto3  # type: ignore[import]
    except ImportError:
        print("ERROR: boto3 is required. Install with: pip install boto3", file=sys.stderr)
        sys.exit(1)
    return boto3.client(
        "s3",
        endpoint_url=f"https://{endpoint}",
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=region,
    )


def _upload(client, local: Path, bucket: str, s3_key: str, dry_run: bool) -> None:
    size = local.stat().st_size
    print(f"  {local}  ({size:,} bytes)")
    print(f"    -> s3://{bucket}/{s3_key}")
    if not dry_run:
        client.upload_file(str(local), bucket, s3_key)
        print(f"    ok")


def main() -> None:
    args = parse_args()

    uploads = [
        (Path(args.csv_path),   f"{_S3_PREFIX}/card_transaction.v1.csv"),
        (Path(args.vocab_path), f"{_S3_PREFIX}/vocab.pkl"),
    ]

    # Validate local files exist before touching S3
    missing = [str(p) for p, _ in uploads if not p.exists()]
    if missing:
        print(
            "ERROR: the following local files do not exist:\n"
            + "\n".join(f"  {f}" for f in missing),
            file=sys.stderr,
        )
        sys.exit(1)

    bucket, endpoint, access, secret = _require_env()

    if args.dry_run:
        print("Dry run — no files will be transferred.\n")
        client = None
    else:
        client = _s3_client(endpoint, access, secret, args.region)

    for local_path, s3_key in uploads:
        _upload(client, local_path, bucket, s3_key, args.dry_run)

    if args.dry_run:
        print("\nDry run complete.")
    else:
        print(
            "\nUpload complete. Training data is ready in S3.\n"
            "You can now submit the PyTorchJob:\n"
            "  oc apply -f openshift/training/pytorchjob-pragma-s.yaml -n pragma-encoder"
        )


if __name__ == "__main__":
    main()
