"""train_pragma() — workbench entry point for PRAGMA pretraining.

Maps the five §2.4 pipeline stages onto a concrete execution path and
returns a PragmaRun for workbench inspection.

Supported modes:
    "dry_run"  — return a PragmaRun immediately without doing any work.
                 Constructs a placeholder DatasetManifest from the requested
                 dataset name and PRAGMAConfig. No S3, no cluster, no subprocess.
                 Useful for previewing pipeline shape and configuration.
    "auto"     — detect environment: OpenShift pod → "cluster"; else → "local".
                 "cluster" and "local" raise NotImplementedError until Component 5.
    "cluster"  — submit a KFTO PyTorchJob via oc apply (not yet implemented).
    "local"    — run scripts/train_pragma.py as a subprocess (not yet implemented).

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

from __future__ import annotations

import os

from src.model.config import PRAGMAConfig
from src.workbench._run import PragmaRun


# ---------------------------------------------------------------------------
# Model size → PRAGMAConfig factory map (case-sensitive per §2.4)
# ---------------------------------------------------------------------------

_MODEL_SIZE_MAP: dict[str, type] = {
    "S": PRAGMAConfig.pragma_s,
    "M": PRAGMAConfig.pragma_m,
    "L": PRAGMAConfig.pragma_l,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_pragma(
    dataset: str,
    model_size: str,
    epochs: int = 10,
    nodes: int = 1,
    prepare_if_missing: bool = True,
    upload: bool = True,
    mode: str = "auto",
) -> PragmaRun:
    """Launch PRAGMA pretraining and return a PragmaRun for inspection.

    Args:
        dataset:            Dataset registry key, e.g. "ibm-tabformer".
                            Must be registered in src/data/adapters/__init__.py.
        model_size:         Model size variant: "S", "M", or "L" (case-sensitive).
                            Maps to PRAGMAConfig.pragma_s/m/l() (Table 1).
        epochs:             Number of pretraining epochs. Default: 10.
        nodes:              Number of training nodes. 1 = single-node (pragma-s.yaml),
                            2 = two-node DDP (pragma-s-2node.yaml). Default: 1.
        prepare_if_missing: If True, run the dataset adapter to prepare and
                            (if upload=True) upload data before training. Default: True.
        upload:             If True, upload prepared artifacts to S3. Set False for
                            local development and tests. Ignored in dry_run. Default: True.
        mode:               Execution mode:
                            "auto"     — detect environment: OpenShift pod → "cluster";
                                         else → "local".
                            "cluster"  — submit a KFTO PyTorchJob via oc apply.
                            "local"    — run scripts/train_pragma.py as a subprocess.
                            "dry_run"  — return a PragmaRun without doing any work;
                                         useful for previewing pipeline shape and config.
                            Default: "auto".

    Returns:
        PragmaRun: wraps the submitted job. Call show_pipeline(), metrics(),
                   artifacts() to inspect progress and results.

    Raises:
        ValueError:      If model_size is not "S", "M", or "L" (case-sensitive).
        KeyError:        If dataset is not registered in the adapter registry.
        FileNotFoundError: If prepare_if_missing=False and dataset data is missing.
    """
    # --- Validate model_size (case-sensitive) before any other work ---
    if model_size not in _MODEL_SIZE_MAP:
        raise ValueError(
            f"model_size must be one of 'S', 'M', 'L' (case-sensitive), "
            f"got {model_size!r}. "
            f"Valid values: {sorted(_MODEL_SIZE_MAP)}"
        )

    # --- Validate dataset via adapter registry (raises KeyError if unknown) ---
    from src.data.adapters import get_adapter
    get_adapter(dataset)  # validate only; raises KeyError if not registered

    # --- Build config from model_size ---
    config: PRAGMAConfig = _MODEL_SIZE_MAP[model_size]()

    # --- Resolve effective mode ---
    effective_mode = mode
    if mode == "auto":
        # Detect environment: presence of KUBERNETES_SERVICE_HOST → OpenShift pod
        effective_mode = "cluster" if os.environ.get("KUBERNETES_SERVICE_HOST") else "local"

    # --- Dispatch ---
    if effective_mode == "dry_run":
        return _build_dry_run(dataset=dataset, config=config)

    raise NotImplementedError(
        f"mode='{effective_mode}' is not yet implemented. "
        f"Use mode='dry_run' to preview the pipeline shape and configuration."
    )


# ---------------------------------------------------------------------------
# Private — dry_run builder
# ---------------------------------------------------------------------------

def _build_dry_run(dataset: str, config: PRAGMAConfig) -> PragmaRun:
    """Return a PragmaRun preview without touching S3, cluster, or adapters.

    Constructs a placeholder DatasetManifest so the returned PragmaRun has a
    valid, inspectable manifest. All steps are in 'pending' state. The run_mode
    field is set to 'dry_run' so show_pipeline() clearly labels this as a preview.

    No side effects: no S3 access, no oc calls, no subprocess, no DatasetAdapter.prepare().
    """
    from src.data.dataset_manifest import DatasetManifest, DatasetShard

    placeholder_shard = DatasetShard(
        uri=f"pragma-encoder/data/{dataset}/[dry-run-placeholder]",
        format="csv",
        rows=None,
    )
    manifest = DatasetManifest(
        dataset_name=dataset,
        dataset_version="dry-run",
        prepared_prefix_uri=f"pragma-encoder/data/{dataset}/",
        shards=(placeholder_shard,),
        vocab_uri=None,
        schema_uri=None,
        manifest_uri=None,
        row_count=None,
        source={"origin": dataset, "mode": "dry_run"},
        config=config,
    )
    return PragmaRun(manifest=manifest, run_mode="dry_run")
