"""Training readiness report — local diagnostic utility (§2.3.5, §2.4).

Provides a human-readable summary of the dataset and model configuration
before a local training run begins. Intended for use at the start of
examples/workbench/06_local_learning_validation.py and any other local
training path.

Reports:
    dataset_name       — which dataset is being used
    dataset_path       — local path to raw data (None if not yet prepared)
    n_sequences        — number of customer histories / sequences
    n_events           — total number of transaction events across all sequences
    key_vocab_size     — number of field-type (key) tokens (§2.2)
    value_vocab_size   — number of value tokens (§2.2, ~28k)
    max_event_tokens   — token budget per event (§2.4, default 24)
    max_events         — maximum events per customer history (§2.4, default 6500)
    mask_rate          — token masking probability (§2.3.5, default 0.15)
    model_variant      — model size identifier ("pragma-s", "pragma-m", "pragma-l")
    trainable_params   — total number of trainable parameters in the model

Design constraints:
    - No boto3, kfp, or kubernetes imports. This module must run locally
      without any cluster credentials or network access.
    - Depends only on src.model.config and src.model.pragma (allowed by
      the dependency graph in DEVELOPMENT_PROCESS.md).
    - Not an nn.Module — pure Python dataclass with no learnable parameters.

Reference: Ostroukhov et al. (2026), Sections 2.2, 2.3.5, 2.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.model.config import PRAGMAConfig
from src.model.pragma import PRAGMA


@dataclass
class TrainingReadinessReport:
    """Summary of dataset and model configuration before a local training run.

    All fields are set by :func:`make_readiness_report`. The report is
    immutable after construction — it is a snapshot of the configuration
    at the time :func:`make_readiness_report` was called.

    Fields:
        dataset_name:     Name of the dataset adapter (e.g. "ibm-tabformer").
        dataset_path:     Local path to the raw data file or directory,
                          or None if the dataset has not been prepared yet.
        n_sequences:      Number of distinct customer histories / sequences.
        n_events:         Total number of transaction events across all sequences.
        key_vocab_size:   Number of field-type (key) vocabulary tokens. §2.2.
        value_vocab_size: Number of value vocabulary tokens. §2.2 (~28k).
        max_event_tokens: Token budget per event (truncation limit). §2.4.
        max_events:       Maximum events per customer history. §2.4.
        mask_rate:        Token masking probability used for MLM. §2.3.5.
        model_variant:    Model size identifier from PRAGMAConfig.model_name.
        trainable_params: Count of requires_grad=True parameters in the model.
    """

    dataset_name:     str
    dataset_path:     Optional[str]
    n_sequences:      int
    n_events:         int
    key_vocab_size:   int
    value_vocab_size: int
    max_event_tokens: int
    max_events:       int
    mask_rate:        float
    model_variant:    str
    trainable_params: int

    def print_report(self) -> None:
        """Print a human-readable training readiness report to stdout.

        Output is clearly labelled as a LOCAL LEARNING VALIDATION report
        so it cannot be mistaken for a production training summary.
        """
        sep = "=" * 60
        print(sep)
        print("  PRAGMA LOCAL LEARNING VALIDATION — Readiness Report")
        print(sep)
        print("  Dataset")
        print(f"    name            : {self.dataset_name}")
        if self.dataset_path is not None:
            print(f"    path            : {self.dataset_path}")
        else:
            print("    path            : (not yet prepared — using synthetic data)")
        print(f"    sequences       : {self.n_sequences:,}")
        print(f"    events (total)  : {self.n_events:,}")
        print()
        print("  Vocabulary (§2.2)")
        print(f"    key_vocab_size  : {self.key_vocab_size:,}")
        print(f"    value_vocab_size: {self.value_vocab_size:,}")
        print()
        print("  Truncation (§2.4)")
        print(f"    max_event_tokens: {self.max_event_tokens}")
        print(f"    max_events      : {self.max_events:,}")
        print()
        print("  Masking (§2.3.5)")
        print(f"    mask_rate       : {self.mask_rate:.2f}  (token masking probability)")
        print()
        print("  Model")
        print(f"    variant         : {self.model_variant}")
        print(f"    trainable params: {self.trainable_params:,}")
        print(sep)


def make_readiness_report(
    config: PRAGMAConfig,
    model: PRAGMA,
    dataset_name: str,
    n_sequences: int,
    n_events: int,
    dataset_path: Optional[str] = None,
) -> TrainingReadinessReport:
    """Build a TrainingReadinessReport from config, model, and dataset metadata.

    Reads vocabulary sizes, truncation limits, and masking probability from
    config. Counts trainable parameters from the model.

    Args:
        config:       PRAGMAConfig — single source of truth for all hyperparameters.
        model:        PRAGMA model instance — parameter count is read from this.
        dataset_name: Human-readable dataset identifier (e.g. "ibm-tabformer").
        n_sequences:  Number of distinct customer histories in the dataset.
        n_events:     Total number of transaction events across all sequences.
        dataset_path: Optional local filesystem path to the raw data file or
                      directory. Pass None when using synthetic or pre-loaded data.

    Returns:
        TrainingReadinessReport — immutable snapshot of the configuration.
    """
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    return TrainingReadinessReport(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        n_sequences=n_sequences,
        n_events=n_events,
        key_vocab_size=config.key_vocab_size,
        value_vocab_size=config.value_vocab_size,
        max_event_tokens=config.max_event_tokens,
        max_events=config.max_events,
        mask_rate=config.token_mask_prob,
        model_variant=config.model_name,
        trainable_params=trainable_params,
    )
