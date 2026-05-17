"""PRAGMA evaluation entrypoint.

Evaluates a pretrained PRAGMA model on downstream financial tasks using
the linear probe protocol from PRAGMA paper Section 3.1.1.

Usage:
    python scripts/evaluate_pragma.py \\
        --checkpoint outputs/pragma-s/checkpoint-final.pt \\
        --task fraud_detection \\
        --data-dir /data/labelled \\
        --output-dir outputs/evaluation

Supported tasks:
    fraud_detection     — Binary, primary metric: AUC (Section 3.2)
    churn_prediction    — Binary, primary metric: AUC (Section 3.3)
    credit_risk         — Binary, primary metric: PR-AUC (Section 3.4)

Reference: Ostroukhov et al. (2026), Section 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PRAGMA, PRAGMAConfig
from src.evaluation import DownstreamEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_TASKS = ["fraud_detection", "churn_prediction", "credit_risk"]
TASK_PRIMARY_METRICS = {
    "fraud_detection": "auc",
    "churn_prediction": "auc",
    "credit_risk": "pr_auc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRAGMA downstream evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PRAGMA checkpoint (.pt file)")
    parser.add_argument("--task", type=str, choices=SUPPORTED_TASKS, required=True,
                        help="Downstream evaluation task")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing labelled evaluation data")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    # TODO: Load config from checkpoint metadata
    config = PRAGMAConfig.pragma_s()
    model = PRAGMA(config).to(device)
    # model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    evaluator = DownstreamEvaluator(
        task_name=args.task,
        metric=TASK_PRIMARY_METRICS[args.task],
    )

    logger.info(f"Evaluating task: {args.task}")
    logger.info("Evaluation loop placeholder — implement data loading.")

    # TODO: Load labelled data, extract embeddings, evaluate.
    # See docs/evaluation.md for the full evaluation protocol.


if __name__ == "__main__":
    main()
