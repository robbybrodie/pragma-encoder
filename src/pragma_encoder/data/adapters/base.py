"""DatasetAdapterProtocol — interface contract for dataset preparation adapters.

Each concrete adapter (IBMTabFormerAdapter, BankTxnAdapter, …) implements this
protocol. The train_pragma() API and tests depend only on this protocol —
not on any concrete adapter class.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pragma_encoder.data.dataset_manifest import DatasetManifest
from pragma_encoder.model.config import PRAGMAConfig


@runtime_checkable
class DatasetAdapterProtocol(Protocol):
    """Interface contract for a PRAGMA dataset preparation adapter.

    An adapter transforms a raw dataset source (CSV, Parquet, LMDB, …) into
    a DatasetManifest. The train_pragma() API consumes only this protocol —
    it never knows whether the data came from IBM TabFormer, bank Parquet, or
    any other format.

    Contract:
        - prepare() returns a DatasetManifest satisfying DatasetManifestProtocol.
        - prepare() validates the source exists before attempting any I/O.
        - prepare(upload=False) must not attempt any S3 connection.
        - dataset_name must match the registry key in adapters/__init__.py._REGISTRY.

    Reference: §2.4 (dataset preparation / data storage pipeline)
    """

    @property
    def dataset_name(self) -> str:
        """Registry key for this adapter, e.g. "ibm-tabformer"."""
        ...

    def prepare(
        self,
        config: PRAGMAConfig,
        upload: bool = True,
    ) -> DatasetManifest:
        """Prepare the dataset and return a manifest describing it.

        Steps:
          1. Validate source data exists (raises FileNotFoundError if not).
          2. Fit FinancialTokenizerPipeline on training records.
          3. Serialise fitted vocab to a local temp file.
          4. If upload=True: upload source data + vocab to S3.
          5. Return DatasetManifest with shard URIs, vocab_uri, config.

        Args:
            config: PRAGMAConfig carrying §2.4 truncation limits. The returned
                    manifest will carry this exact config instance.
            upload: If True, upload prepared artifacts to S3. Set False for
                    local development and tests (no S3 connection made).

        Returns:
            DatasetManifest: immutable reference to the prepared dataset.

        Raises:
            FileNotFoundError: If the source data file does not exist.
        """
        ...
