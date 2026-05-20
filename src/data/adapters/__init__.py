"""Dataset adapters — transform raw dataset sources into DatasetManifest objects.

Each adapter handles one dataset source format. The adapter:
  1. Validates the source (CSV, Parquet, LMDB, …) exists and is readable.
  2. Fits the FinancialTokenizerPipeline on training records.
  3. Uploads prepared artifacts to S3 (optional — skip for local dev/test).
  4. Returns a DatasetManifest describing the prepared dataset.

Registered adapters:
    "ibm-tabformer"       → IBMTabFormerAdapter       (IBM TabFormer CSV; requires real data + S3)
    "ibm-tabformer-smoke" → IBMTabFormerSmokeAdapter  (synthetic 15-row CSV; no S3; KFP smoke only)

Adding a new adapter (e.g. bank transaction Parquet):
    1. Create src/data/adapters/bank_txn.py implementing DatasetAdapterProtocol.
    2. Register the key in _REGISTRY below.
    3. Write tests/test_dataset_adapters.py tests for the new adapter.
    No other files need to change.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

from .base import DatasetAdapterProtocol
from .ibm_tabformer import IBMTabFormerAdapter
from .ibm_tabformer_smoke import IBMTabFormerSmokeAdapter

_REGISTRY: dict[str, type] = {
    "ibm-tabformer": IBMTabFormerAdapter,
    "ibm-tabformer-smoke": IBMTabFormerSmokeAdapter,
}


def get_adapter(dataset_name: str) -> type:
    """Return the adapter class for the given dataset registry key.

    Args:
        dataset_name: Registry key, e.g. "ibm-tabformer".

    Raises:
        KeyError: If the dataset_name is not registered.
    """
    if dataset_name not in _REGISTRY:
        raise KeyError(
            f"No adapter registered for dataset '{dataset_name}'. "
            f"Registered adapters: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[dataset_name]


__all__ = ["DatasetAdapterProtocol", "IBMTabFormerAdapter", "IBMTabFormerSmokeAdapter", "get_adapter"]
