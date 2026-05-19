"""KFP SDK v2 pipeline components for PRAGMA training.

Five components implementing the §2.4 training infrastructure stages:

  Stage 1 — prepare_dataset   : Prepare dataset via DatasetAdapter (fit tokeniser)
  Stage 2 — upload_artifacts  : Upload prepared artifacts to S3 (idempotent)
  Stage 3 — submit_pytorchjob : Configure and submit KFTO PyTorchJob
  Stage 4 — run_pretraining   : Execute pretraining (MEM objective §2.3.5)
  Stage 5 — export_checkpoint : Upload model checkpoints and outputs to S3

KFP is an optional dependency.  When kfp is not installed, the _KFP_AVAILABLE
flag is False and @dsl.component is not applied — components are plain Python
callables that can be imported and inspected without a running KFP server.

Design (ADR 003):
  - pipeline/ → src/ only.  No circular dependency introduced.
  - DatasetManifest / manifest_uri is the canonical dataset contract.
  - No PVC-backed dataset storage.  All data lives in S3.
  - prepare_dataset delegates to DatasetAdapter (get_adapter).
  - run_pretraining uses the same model_size → PRAGMAConfig mapping as
    train_pragma() so config is defined in exactly one place.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

# ---------------------------------------------------------------------------
# KFP optional import guard
# ---------------------------------------------------------------------------

try:
    from kfp import dsl as _dsl
    _KFP_AVAILABLE: bool = True
except ImportError:
    _dsl = None
    _KFP_AVAILABLE: bool = False


def _component(**kwargs):
    """Return @dsl.component decorator if KFP available; identity decorator otherwise."""
    if _KFP_AVAILABLE:
        return _dsl.component(**kwargs)
    return lambda fn: fn


# ---------------------------------------------------------------------------
# The five §2.4 pipeline stage names
# Must mirror PIPELINE_STEP_NAMES in src/workbench/_run.py (ADR 003).
# ---------------------------------------------------------------------------

PIPELINE_STAGE_NAMES: tuple[str, ...] = (
    "prepare",   # Prepare dataset via DatasetAdapter; return manifest URI
    "upload",    # Upload prepared artifacts to S3 (idempotent)
    "submit",    # Configure and submit KFTO PyTorchJob
    "train",     # Execute pretraining — masked event modelling (§2.3.5)
    "export",    # Upload model checkpoints and outputs to S3
)

_BASE_IMAGE = (
    "image-registry.openshift-image-registry.svc:5000"
    "/pragma-encoder/pragma-encoder-workbench:latest"
)


# ---------------------------------------------------------------------------
# Stage 1 — prepare_dataset
# ---------------------------------------------------------------------------

@_component(base_image=_BASE_IMAGE)
def prepare_dataset(
    dataset_name: str,
    model_size: str,
    upload: bool = False,
) -> str:
    """Prepare a dataset via the DatasetAdapter registry.

    Resolves the adapter for dataset_name via get_adapter(), builds the
    PRAGMAConfig for model_size, runs DatasetAdapter.prepare(), and returns
    the prepared_prefix_uri from the resulting DatasetManifest.

    Does NOT hardcode IBM TabFormer logic — any registered adapter is
    supported.  New dataset adapters register in src/data/adapters/__init__.py
    and require no changes to this component.

    Args:
        dataset_name: Adapter registry key, e.g. "ibm-tabformer".
        model_size:   Model variant "S", "M", or "L" (case-sensitive).
                      Determines PRAGMAConfig truncation limits passed to
                      the adapter (max_event_tokens=24, max_profile_tokens=200,
                      max_events=6500).
        upload:       If True, upload prepared artifacts to S3 during prepare.
                      Default: False — S3 upload is the upload_artifacts stage.

    Returns:
        prepared_prefix_uri from the returned DatasetManifest (S3 prefix URI).

    Raises:
        KeyError:          If dataset_name is not in the adapter registry.
        ValueError:        If model_size is not "S", "M", or "L".
        FileNotFoundError: If source data is missing.
    """
    from src.data.adapters import get_adapter
    from src.model.config import PRAGMAConfig

    _config_map = {
        "S": PRAGMAConfig.pragma_s,
        "M": PRAGMAConfig.pragma_m,
        "L": PRAGMAConfig.pragma_l,
    }
    if model_size not in _config_map:
        raise ValueError(
            f"model_size must be 'S', 'M', or 'L' (case-sensitive), "
            f"got {model_size!r}"
        )
    config = _config_map[model_size]()
    adapter_cls = get_adapter(dataset_name)
    adapter = adapter_cls()
    manifest = adapter.prepare(config, upload=upload)
    return manifest.prepared_prefix_uri


# ---------------------------------------------------------------------------
# Stage 2 — upload_artifacts
# ---------------------------------------------------------------------------

@_component(base_image=_BASE_IMAGE)
def upload_artifacts(
    manifest_uri: str,
    bucket: str = "",
) -> str:
    """Upload prepared artifacts to S3 (idempotent).

    Reads the DatasetManifest at manifest_uri and uploads all shards and the
    vocabulary file to S3.  Idempotent: re-running does not corrupt existing
    data.

    S3 credentials come from the pragma-workbench-env Secret only
    (openshift-storage-pattern.md rule 4).

    Args:
        manifest_uri: S3 prefix URI returned by prepare_dataset.
        bucket:       Override S3 bucket name.  Defaults to the value of
                      MODEL_REGISTRY_BUCKET environment variable.

    Returns:
        manifest_uri: The same manifest_uri (passed to downstream stages).
    """
    raise NotImplementedError(
        "upload_artifacts: upload prepared artifacts from manifest_uri to S3. "
        "S3 credentials from pragma-workbench-env Secret "
        "(MODEL_REGISTRY_ENDPOINT_URL / MODEL_REGISTRY_BUCKET)."
    )


# ---------------------------------------------------------------------------
# Stage 3 — submit_pytorchjob
# ---------------------------------------------------------------------------

@_component(base_image=_BASE_IMAGE)
def submit_pytorchjob(
    manifest_uri: str,
    model_size: str = "S",
    nodes: int = 1,
    epochs: int = 10,
    namespace: str = "pragma-encoder",
) -> str:
    """Configure and submit a KFTO PyTorchJob to OpenShift AI.

    Logical/configuration stage — no real cluster mutation in unit tests.

    Selects the correct PyTorchJob manifest based on nodes:
        nodes=1 → openshift/training/pytorchjob-pragma-s.yaml
        nodes=2 → openshift/training/pytorchjob-pragma-s-2node.yaml (ADR 003)

    No PVC-backed canonical dataset storage is introduced.  Data is sourced
    from S3 via the init container (openshift-storage-pattern.md).

    Args:
        manifest_uri: DatasetManifest URI — passed as DATA_URI env var to job.
        model_size:   Model variant "S", "M", or "L".  Default: "S".
        nodes:        Number of training nodes.  1 = single-node,
                      2 = two-node DDP.  Default: 1.
        epochs:       Number of pretraining epochs.  Default: 10.
        namespace:    OpenShift namespace for the PyTorchJob.
                      Default: "pragma-encoder".

    Returns:
        job_name: The submitted PyTorchJob name (for monitoring).
    """
    raise NotImplementedError(
        "submit_pytorchjob: apply the correct PyTorchJob manifest via oc apply. "
        "nodes=1 → pytorchjob-pragma-s.yaml; "
        "nodes=2 → pytorchjob-pragma-s-2node.yaml."
    )


# ---------------------------------------------------------------------------
# Stage 4 — run_pretraining
# ---------------------------------------------------------------------------

@_component(base_image=_BASE_IMAGE)
def run_pretraining(
    manifest_uri: str,
    model_size: str = "S",
    nodes: int = 1,
    epochs: int = 10,
) -> str:
    """Execute PRAGMA pretraining and return the checkpoint URI.

    Monitors the PyTorchJob submitted in submit_pytorchjob until completion,
    then returns the S3 URI of the final checkpoint.

    Uses the same model_size → PRAGMAConfig mapping as train_pragma() in
    src/workbench/_api.py so configuration is defined in exactly one place.
    Specifically, the same _config_map {"S": PRAGMAConfig.pragma_s, ...}
    is used to ensure consistency.

    Args:
        manifest_uri: DatasetManifest URI for the prepared training dataset.
                      Canonical §2.4 training contract — not a raw file path.
        model_size:   Model variant "S", "M", or "L".  Maps to PRAGMAConfig.
        nodes:        Number of training nodes (1 or 2).
        epochs:       Number of pretraining epochs.

    Returns:
        checkpoint_uri: S3 URI of the final model checkpoint.
    """
    from src.model.config import PRAGMAConfig

    # Same model_size → PRAGMAConfig mapping as train_pragma()
    # (src/workbench/_api.py::_MODEL_SIZE_MAP) — no second implementation.
    _config_map = {
        "S": PRAGMAConfig.pragma_s,
        "M": PRAGMAConfig.pragma_m,
        "L": PRAGMAConfig.pragma_l,
    }
    if model_size not in _config_map:
        raise ValueError(
            f"model_size must be 'S', 'M', or 'L', got {model_size!r}"
        )
    raise NotImplementedError(
        "run_pretraining: monitor PyTorchJob and return checkpoint URI. "
        "Working training entrypoint: scripts/train_pragma.py."
    )


# ---------------------------------------------------------------------------
# Stage 5 — export_checkpoint
# ---------------------------------------------------------------------------

@_component(base_image=_BASE_IMAGE)
def export_checkpoint(
    checkpoint_uri: str,
    model_size: str = "S",
    export_prefix: str = "pragma-encoder/checkpoints/",
) -> str:
    """Export model checkpoints and vocabulary to S3.

    Copies the final checkpoint to the canonical S3 export prefix.
    S3 path format follows openshift-storage-pattern.md:
        pragma-encoder/checkpoints/pragma-s/checkpoint_epoch<NNNN>.pt

    Args:
        checkpoint_uri:  S3 URI of the final checkpoint from run_pretraining.
        model_size:      Model variant ("S", "M", "L").  Sets the S3 key prefix.
        export_prefix:   Override S3 export prefix.
                         Default: "pragma-encoder/checkpoints/".

    Returns:
        export_uri: S3 URI of the exported model directory.
    """
    raise NotImplementedError(
        "export_checkpoint: copy checkpoint to S3 canonical export prefix "
        "and return the export URI. "
        "Format: pragma-encoder/checkpoints/pragma-{s,m,l}/checkpoint_epoch<NNNN>.pt"
    )
