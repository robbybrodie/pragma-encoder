"""KFP SDK v2 pipeline components for PRAGMA training.

Five components implementing the sec.2.4 training infrastructure stages:

  Stage 1 - prepare_dataset   : Prepare dataset via DatasetAdapter (fit tokeniser)
  Stage 2 - upload_artifacts  : Upload prepared artifacts to S3 (idempotent)
  Stage 3 - submit_pytorchjob : Configure and submit KFTO PyTorchJob
  Stage 4 - run_pretraining   : Execute pretraining (MEM objective sec.2.3.5)
  Stage 5 - export_checkpoint : Upload model checkpoints and outputs to S3

KFP is an optional dependency.  When kfp is not installed, the _KFP_AVAILABLE
flag is False and @dsl.component is not applied - components are plain Python
callables that can be imported and inspected without a running KFP server.

Design (ADR 003):
  - pipeline/ imports from the pragma_encoder wheel only.  No circular
    dependency introduced.  No source-tree path assumptions.
  - DatasetManifest / manifest_uri is the canonical dataset contract.
  - No PVC-backed dataset storage.  All data lives in S3.
  - prepare_dataset delegates to DatasetAdapter (get_adapter).
  - run_pretraining uses the same model_size -> PRAGMAConfig mapping as
    train_pragma() so config is defined in exactly one place.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

# ---------------------------------------------------------------------------
# KFP optional import guard
# ---------------------------------------------------------------------------

import os

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
# The five sec.2.4 pipeline stage names
# Must mirror PIPELINE_STEP_NAMES in
# tools/workbench/_run.py (ADR 003).
# ---------------------------------------------------------------------------

PIPELINE_STAGE_NAMES: tuple[str, ...] = (
    "prepare",   # Prepare dataset via DatasetAdapter; return manifest URI
    "upload",    # Upload prepared artifacts to S3 (idempotent)
    "submit",    # Configure and submit KFTO PyTorchJob
    "train",     # Execute pretraining - masked event modelling (sec.2.3.5)
    "export",    # Upload model checkpoints and outputs to S3
)

# ---------------------------------------------------------------------------
# Component base image
#
# KFP component pods do NOT inherit the workbench pod's environment.
# The component image must have the pragma_encoder wheel installed and all
# Python dependencies present.  The PRAGMA training image is the correct
# choice — it is built from openshift/training/Dockerfile.training with
# the pragma_encoder wheel installed via pip.
#
# Override at compile time via env var (read once at module import):
#   PRAGMA_KFP_COMPONENT_IMAGE — explicit override (highest priority)
#   PRAGMA_TRAINING_IMAGE      — training image URI (fallback)
#
# Do NOT use the workbench image (pragma-encoder-workbench) — it does not
# have pragma_encoder installed as a production package.  KFP component
# pods would fail with:
#   ModuleNotFoundError: No module named 'pragma_encoder'
#
# Do NOT use runtime source cloning — not the default pattern.
# ---------------------------------------------------------------------------

_DEFAULT_COMPONENT_IMAGE = (
    "image-registry.openshift-image-registry.svc:5000"
    "/pragma-encoder/pragma-encoder-training:latest"
)

# Read env var at import time — KFP @dsl.component captures base_image
# at decoration time, so the env var must be set before this module is imported.
_BASE_IMAGE: str = (
    os.environ.get("PRAGMA_KFP_COMPONENT_IMAGE")
    or os.environ.get("PRAGMA_TRAINING_IMAGE")
    or _DEFAULT_COMPONENT_IMAGE
)


# ---------------------------------------------------------------------------
# Stage 1 - prepare_dataset
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

    Does NOT hardcode IBM TabFormer logic - any registered adapter is
    supported.  New dataset adapters register in
    pragma_encoder/data/adapters/__init__.py and require no changes to this
    component.

    Args:
        dataset_name: Adapter registry key, e.g. "ibm-tabformer".
        model_size:   Model variant "S", "M", or "L" (case-sensitive).
                      Determines PRAGMAConfig truncation limits passed to
                      the adapter (max_event_tokens=24, max_profile_tokens=200,
                      max_events=6500).
        upload:       If True, upload prepared artifacts to S3 during prepare.
                      Default: False - S3 upload is the upload_artifacts stage.

    Returns:
        prepared_prefix_uri from the returned DatasetManifest (S3 prefix URI).

    Raises:
        KeyError:          If dataset_name is not in the adapter registry.
        ValueError:        If model_size is not "S", "M", or "L".
        FileNotFoundError: If source data is missing.
    """
    from pragma_encoder.data.adapters import get_adapter
    from pragma_encoder.model.config import PRAGMAConfig

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
# Stage 2 - upload_artifacts
# ---------------------------------------------------------------------------

@_component(base_image=_BASE_IMAGE)
def upload_artifacts(
    manifest_uri: str,
    bucket: str = "",
) -> str:
    """Upload prepared artifacts to S3 (idempotent).

    If manifest_uri is already an S3 URI (starts with 's3://'), the artifacts
    are assumed to be in place — returns manifest_uri unchanged.

    If manifest_uri is a local path (e.g. from ibm-tabformer-smoke), there is
    nothing durable to upload; returns manifest_uri unchanged.  Full S3 upload
    (reading shards from manifest, uploading via boto3) is a future milestone.

    S3 credentials come from AWS_* env vars supplied by the
    OpenShift AI Connection (test fixture/default: pragma-workbench-env Secret).

    Args:
        manifest_uri: Prefix URI returned by prepare_dataset (S3 or local).
        bucket:       Override S3 bucket name.  Defaults to the value of
                      AWS_S3_BUCKET environment variable.

    Returns:
        manifest_uri: The same manifest_uri (passed to downstream stages).
    """
    # Pass-through: downstream stages receive the same manifest_uri.
    # Full S3 upload implementation is a future milestone (see docs/tech-debt.md).
    return manifest_uri


# ---------------------------------------------------------------------------
# Stage 3 - submit_pytorchjob
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

    Logical/configuration stage - no real cluster mutation in unit tests.

    Selects the correct PyTorchJob manifest based on nodes:
        nodes=1 -> openshift/training/pytorchjob-pragma-s.yaml
        nodes=2 -> openshift/training/pytorchjob-pragma-s-2node.yaml (ADR 003)

    No PVC-backed canonical dataset storage is introduced.  Data is sourced
    from S3 via the init container (openshift-storage-pattern.md).

    Args:
        manifest_uri: DatasetManifest URI - passed as DATA_URI env var to job.
        model_size:   Model variant "S", "M", or "L".  Default: "S".
        nodes:        Number of training nodes.  1 = single-node,
                      2 = two-node DDP.  Default: 1.
        epochs:       Number of pretraining epochs.  Default: 10.
        namespace:    OpenShift namespace for the PyTorchJob.
                      Default: "pragma-encoder".

    Returns:
        job_name: The submitted PyTorchJob name (for monitoring).
    """
    import os as _os
    # Smoke stub: return a synthetic job name.
    # Real implementation: apply pytorchjob-pragma-{s,m,l}.yaml via kubernetes API.
    # nodes=1 -> pytorchjob-pragma-s.yaml; nodes=2 -> pytorchjob-pragma-s-2node.yaml.
    # PyTorchJob submission is a future milestone — do not implement here.
    _run_id = _os.environ.get("KFP_RUN_ID", "smoke")
    return f"pragma-{model_size.lower()}-{_run_id[:8]}"


# ---------------------------------------------------------------------------
# Stage 4 - run_pretraining
# ---------------------------------------------------------------------------

@_component(base_image=_BASE_IMAGE)
def run_pretraining(
    manifest_uri: str,
    model_size: str = "S",
    nodes: int = 1,
    epochs: int = 10,
    max_steps: int = 0,
    limit_rows: int = 0,
    csv_uri: str = "",
    vocab_uri: str = "",
    output_prefix: str = "pragma-encoder/checkpoints/",
    scratch_dir: str = "/tmp/pragma-run",
) -> str:
    """Execute PRAGMA pretraining via the wheel-based training entrypoint.

    Downloads dataset and vocab from S3 (when S3 env vars are present and URIs
    are S3 keys) or uses local paths, then runs ``pragma-encoder-train``.
    Writes checkpoint(s) and metadata.json to a job-local scratch directory,
    then uploads the final checkpoint and metadata.json to S3 under
    ``output_prefix``.

    Data injection contract:
      - ``csv_uri`` / ``vocab_uri`` are S3 keys when they do not start with
        ``/`` (i.e. not absolute local paths). S3 credentials come from
        native OpenShift AI Connection env vars (AWS_S3_BUCKET, AWS_S3_ENDPOINT,
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).
      - When ``csv_uri`` / ``vocab_uri`` start with ``/`` they are treated as
        local paths (no S3 download). This supports the smoke adapter whose
        ``manifest_uri`` is a local tmpdir path.
      - Default URIs are derived from ``manifest_uri`` using IBM TabFormer
        S3 path conventions: ``{manifest_uri}card_transaction.v1.csv`` and
        ``{manifest_uri}vocab.pkl``.

    Scratch/staging contract:
      - All downloaded input files land under ``scratch_dir`` (default:
        ``/tmp/pragma-run``). This is a KFP component pod-local emptyDir
        equivalent — ephemeral, not shared, not a PVC.
      - ``scratch_dir`` is configurable so tests can redirect staging to
        a tmpdir and so future confidential runtimes can restrict staging
        to attested memory regions.
      - Training output (checkpoints, metadata.json) also lands under
        ``scratch_dir/output`` and is uploaded to S3 before the component exits.
      - Raw data is not baked into the image, wheel, or Git.

    Confidential-computing seam:
      No HardwareProfile or Connection names are referenced here. The wheel
      reads native AWS_* env vars from the process environment — it does not
      know which Kubernetes Secret or OpenShift AI Connection provided them.
      The same env vars can be injected via attestation-based secret release
      (Trustee / CoCo) without changing this component.

    Args:
        manifest_uri:   DatasetManifest prefix URI from prepare_dataset.
                        Used to derive csv_uri/vocab_uri when not provided.
        model_size:     Model variant "S", "M", or "L".  Maps to PRAGMAConfig.
        nodes:          Number of training nodes (1 or 2).
        epochs:         Number of pretraining epochs.
        max_steps:      Stop after this many training steps (0 = train all epochs).
        limit_rows:     Cap dataset to this many customers (0 = all rows).
                        Use for smoke/tiny runs that must produce artifacts quickly.
        csv_uri:        Explicit CSV path or S3 key. Defaults to manifest_uri +
                        card_transaction.v1.csv (IBM TabFormer S3 convention).
        vocab_uri:      Explicit vocab path or S3 key. Defaults to manifest_uri +
                        vocab.pkl.
        output_prefix:  S3 key prefix for checkpoint export.
                        Default: "pragma-encoder/checkpoints/".
        scratch_dir:    Job-local directory for staging downloaded inputs and
                        training outputs. Must be pod-local/ephemeral (not a
                        shared PVC). Default: "/tmp/pragma-run".

    Returns:
        checkpoint_uri: Local path or S3 key of the final model checkpoint.
    """
    import os as _os
    import pathlib
    import subprocess
    import sys

    # ---- Resolve CSV and vocab URIs from manifest_uri if not explicit -----
    _prefix = manifest_uri.rstrip("/") + "/"
    _csv_uri  = csv_uri  if csv_uri  else _prefix + "card_transaction.v1.csv"
    _vocab_uri = vocab_uri if vocab_uri else _prefix + "vocab.pkl"

    # ---- Determine whether URIs are local or S3 keys ----------------------
    # Local path: starts with "/" (absolute) or "./" (relative).
    # S3 key: everything else (e.g. "pragma-encoder/data/tabformer/...").
    def _is_local(uri: str) -> bool:
        return uri.startswith("/") or uri.startswith("./") or uri.startswith("../")

    # ---- Job-local scratch directory (configurable, ephemeral) ------------
    _work_dir = pathlib.Path(scratch_dir)
    _work_dir.mkdir(parents=True, exist_ok=True)

    if _is_local(_csv_uri):
        local_csv  = pathlib.Path(_csv_uri)
        local_vocab = pathlib.Path(_vocab_uri)
    else:
        # S3 download using native AWS_* env vars (OpenShift AI Connection).
        # Staging lands in the job-local scratch_dir — not a shared PVC.
        _bucket   = _os.environ.get("AWS_S3_BUCKET", "").strip()
        _endpoint = _os.environ.get("AWS_S3_ENDPOINT", "").strip()
        if not _bucket or not _endpoint:
            raise EnvironmentError(
                "run_pretraining: S3 URI provided but AWS_S3_BUCKET or "
                "AWS_S3_ENDPOINT is not set. Supply an OpenShift AI S3 "
                "Connection via the workbench environment secret, or pass "
                "local csv_uri / vocab_uri paths."
            )
        import boto3 as _boto3  # noqa: PLC0415
        _ep_url = _endpoint if _endpoint.startswith("http") else f"https://{_endpoint}"
        _s3 = _boto3.client(
            "s3",
            endpoint_url=_ep_url,
            aws_access_key_id=_os.environ.get("AWS_ACCESS_KEY_ID", "") or None,
            aws_secret_access_key=_os.environ.get("AWS_SECRET_ACCESS_KEY", "") or None,
        )
        local_csv   = _work_dir / pathlib.Path(_csv_uri).name
        local_vocab = _work_dir / "vocab.pkl"
        print(f"[run_pretraining] Downloading CSV from s3://{_bucket}/{_csv_uri} ...")
        _s3.download_file(_bucket, _csv_uri, str(local_csv))
        print(f"[run_pretraining] Downloading vocab from s3://{_bucket}/{_vocab_uri} ...")
        _s3.download_file(_bucket, _vocab_uri, str(local_vocab))

    # ---- Validate downloaded / local paths ---------------------------------
    if not local_csv.exists():
        raise FileNotFoundError(
            f"[run_pretraining] CSV not found: {local_csv}. "
            "Check csv_uri or manifest_uri."
        )
    if not local_vocab.exists():
        raise FileNotFoundError(
            f"[run_pretraining] Vocab not found: {local_vocab}. "
            "Fit the tokenizer first (prepare_dataset stage)."
        )

    # ---- Run training entrypoint -------------------------------------------
    _output_dir = _work_dir / "output"
    _output_dir.mkdir(parents=True, exist_ok=True)

    _model_variant_map = {"S": "pragma-s", "M": "pragma-m", "L": "pragma-l"}
    if model_size not in _model_variant_map:
        raise ValueError(f"model_size must be 'S', 'M', or 'L', got {model_size!r}")
    _model_variant = _model_variant_map[model_size]

    # --dataset-name records the original dataset reference in metadata.json.
    # For S3 runs: the S3 key is the reference. For local runs: empty (csv-path is used).
    _dataset_name = _csv_uri if not _is_local(_csv_uri) else ""

    _cmd = [
        sys.executable, "-m", "pragma_encoder.training.train",
        "--csv-path",      str(local_csv),
        "--vocab-path",    str(local_vocab),
        "--output-dir",    str(_output_dir),
        "--model-variant", _model_variant,
        "--epochs",        str(epochs),
        "--num-workers",   "0",
        "--batch-size",    "1",
    ]
    if max_steps > 0:
        _cmd.extend(["--max-steps", str(max_steps)])
    if limit_rows > 0:
        _cmd.extend(["--limit-rows", str(limit_rows)])
    if _dataset_name:
        _cmd.extend(["--dataset-name", _dataset_name])

    print(f"[run_pretraining] Running: {' '.join(_cmd)}")
    subprocess.run(_cmd, check=True)

    # ---- Find final checkpoint and metadata --------------------------------
    _ckpt_files = sorted(_output_dir.glob("checkpoint_epoch*.pt"))
    if not _ckpt_files:
        raise RuntimeError(
            "[run_pretraining] Training completed but no checkpoint found in "
            f"{_output_dir}. Check training logs above."
        )
    _final_ckpt = _ckpt_files[-1]
    _meta_path = _output_dir / "metadata.json"
    print(f"[run_pretraining] Checkpoint produced: {_final_ckpt}")
    print(f"[run_pretraining] metadata.json present: {_meta_path.exists()}")

    # ---- Upload checkpoint and metadata to S3 if configured ---------------
    _bucket = _os.environ.get("AWS_S3_BUCKET", "").strip()
    _endpoint = _os.environ.get("AWS_S3_ENDPOINT", "").strip()
    if _bucket and _endpoint and not _is_local(_csv_uri):
        _ep_url = _endpoint if _endpoint.startswith("http") else f"https://{_endpoint}"
        import boto3 as _boto3  # noqa: PLC0415  # already imported above if S3 used
        _s3 = _boto3.client(
            "s3",
            endpoint_url=_ep_url,
            aws_access_key_id=_os.environ.get("AWS_ACCESS_KEY_ID", "") or None,
            aws_secret_access_key=_os.environ.get("AWS_SECRET_ACCESS_KEY", "") or None,
        )
        _s3_key = f"{output_prefix.rstrip('/')}/{_model_variant}/{_final_ckpt.name}"
        _s3.upload_file(str(_final_ckpt), _bucket, _s3_key)
        print(f"[run_pretraining] Checkpoint uploaded to s3://{_bucket}/{_s3_key}")
        # Upload metadata.json alongside the checkpoint
        if _meta_path.exists():
            _meta_s3_key = f"{output_prefix.rstrip('/')}/{_model_variant}/metadata.json"
            _s3.upload_file(str(_meta_path), _bucket, _meta_s3_key)
            print(f"[run_pretraining] metadata.json uploaded to s3://{_bucket}/{_meta_s3_key}")
        return _s3_key

    return str(_final_ckpt)


# ---------------------------------------------------------------------------
# Stage 5 - export_checkpoint
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
    # Smoke stub: return a synthetic export URI.
    # Real implementation: copy checkpoint_uri to S3 canonical export prefix via boto3.
    # Format: pragma-encoder/checkpoints/pragma-{s,m,l}/checkpoint_epoch<NNNN>.pt
    # S3 export is a future milestone — do not implement here.
    return f"{export_prefix}pragma-{model_size.lower()}/{checkpoint_uri.rsplit('/', 1)[-1]}"
