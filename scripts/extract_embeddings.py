"""PRAGMA embedding extraction script.

Extracts contextualised event embeddings from a pretrained PRAGMA model
for use in downstream tasks. Outputs embeddings as numpy arrays or parquet.

The extracted embeddings correspond to the History Encoder output at each
event position, as described in Section 3.1 of the PRAGMA paper.

Usage:
    python scripts/extract_embeddings.py \\
        --checkpoint outputs/pragma-s/checkpoint-final.pt \\
        --data-dir /data/transactions \\
        --output-dir /data/embeddings \\
        --pooling usr

Pooling strategies:
    usr  — zh[:,0,:]              — [USR] token (user-level representation)
    last — zh[:,-1,:]             — final [EVT] token (most recent event)
    mean — zh[:,1:,:].mean(dim=1) — mean of all [EVT] tokens

Reference: Ostroukhov et al. (2026), Section 3.1
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pragma_encoder.model import PRAGMA, PRAGMAConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PRAGMA embeddings")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/embeddings")
    parser.add_argument("--pooling", type=str, default="usr",
                        choices=["usr", "mean", "last"],
                        help="usr=zh[:,0,:] | last=zh[:,-1,:] | mean=mean of [EVT] tokens")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--format", type=str, default="numpy",
                        choices=["numpy", "parquet"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    config = PRAGMAConfig.pragma_s()
    model = PRAGMA(config).to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting embeddings with pooling='{args.pooling}'")
    logger.info("Extraction loop placeholder — implement data loading.")

    # TODO: Implement DataLoader, forward pass, and embedding saving.
    # See notebooks/03_pragma_embedding_probe.ipynb for interactive usage.


if __name__ == "__main__":
    main()
