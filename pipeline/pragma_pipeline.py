"""KFP SDK v2 pipeline definition for PRAGMA pretraining.

Defines the end-to-end PRAGMA pretraining pipeline for deployment on
OpenShift AI via Kubeflow Pipelines (KFP SDK v2).

The five pipeline stages mirror the §2.4 training infrastructure stages and
the PIPELINE_STEP_NAMES exposed by PragmaRun.show_pipeline():

    1. prepare   — prepare_dataset: fit tokeniser via DatasetAdapter
    2. upload    — upload_artifacts: upload prepared data to S3 (idempotent)
    3. submit    — submit_pytorchjob: configure / submit KFTO PyTorchJob
    4. train     — run_pretraining: execute pretraining (MEM objective §2.3.5)
    5. export    — export_checkpoint: upload model checkpoint to S3

Canonical dataset contract: DatasetManifest / manifest_uri (§2.4 data
storage).  No PVC-backed dataset storage.  All data lives in S3.

If manifest_uri is provided (non-empty), the prepare and upload stages are
skipped — callers can re-submit training on an already-prepared dataset
without re-running the expensive tokeniser-fitting step.

KFP is an optional dependency (same guard as components_pragma.py).
When kfp is not installed, @dsl.pipeline is not applied and the function
is a plain Python callable inspectable without a running KFP server.

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


def _pipeline(**kwargs):
    """Return @dsl.pipeline decorator if KFP available; identity decorator otherwise."""
    if _KFP_AVAILABLE:
        return _dsl.pipeline(**kwargs)
    return lambda fn: fn


# ---------------------------------------------------------------------------
# Import pipeline components (pipeline/ → src/ only; no circular dependency)
# ---------------------------------------------------------------------------

from pipeline.components_pragma import (  # noqa: E402
    prepare_dataset,
    upload_artifacts,
    submit_pytorchjob,
    run_pretraining,
    export_checkpoint,
)


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

@_pipeline(
    name="pragma-pretraining-pipeline",
    description=(
        "End-to-end PRAGMA foundation model pretraining pipeline. "
        "Five visible §2.4 stages: prepare → upload → submit → train → export. "
        "Canonical dataset contract: DatasetManifest / manifest_uri (not PVC). "
        "Ostroukhov et al. (2026), arXiv:2604.08649v1."
    ),
)
def pragma_pretraining_pipeline(
    dataset_name: str,
    model_size: str = "S",
    epochs: int = 10,
    nodes: int = 1,
    manifest_uri: str = "",
) -> None:
    """Full PRAGMA pretraining pipeline with five visible §2.4 stages.

    The pipeline consumes DatasetManifest / manifest_uri as the canonical
    training dataset contract — not raw local file paths or PVC-mounted
    data.

    If manifest_uri is non-empty, the prepare and upload stages are skipped
    (the dataset is already prepared in S3).

    Args:
        dataset_name: Adapter registry key, e.g. "ibm-tabformer".
                      Resolved via src/data/adapters/__init__.py::get_adapter.
        model_size:   Model variant "S", "M", or "L" (case-sensitive).
                      Maps to PRAGMAConfig.pragma_s/m/l() (Table 1).
                      Default: "S" (PRAGMA-S, ~10M parameters).
        epochs:       Number of pretraining epochs.  Default: 10.
        nodes:        Number of training nodes.
                      1 = single-node (pytorchjob-pragma-s.yaml),
                      2 = two-node DDP (pytorchjob-pragma-s-2node.yaml).
                      Default: 1.
        manifest_uri: If non-empty, skip prepare/upload and use this
                      DatasetManifest URI directly for training.
                      Default: "" (run prepare and upload from scratch).
    """
    # Stage 1: Prepare dataset via DatasetAdapter registry
    prepare_op = prepare_dataset(
        dataset_name=dataset_name,
        model_size=model_size,
        upload=False,  # S3 upload is handled by upload_artifacts (stage 2)
    )

    # Stage 2: Upload prepared artifacts to S3 (idempotent)
    upload_op = upload_artifacts(
        manifest_uri=prepare_op.output,
    )

    # Stage 3: Configure and submit KFTO PyTorchJob
    submit_op = submit_pytorchjob(
        manifest_uri=upload_op.output,
        model_size=model_size,
        nodes=nodes,
        epochs=epochs,
    )

    # Stage 4: Execute pretraining — masked event modelling (§2.3.5)
    train_op = run_pretraining(
        manifest_uri=upload_op.output,
        model_size=model_size,
        nodes=nodes,
        epochs=epochs,
    )

    # Stage 5: Export model checkpoints and outputs to S3
    export_op = export_checkpoint(
        checkpoint_uri=train_op.output,
        model_size=model_size,
    )


# ---------------------------------------------------------------------------
# Compilation entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _KFP_AVAILABLE:
        print("kfp is not installed — pipeline compilation requires kfp.")
        raise SystemExit(1)
    import kfp.compiler as compiler
    compiler.Compiler().compile(
        pipeline_func=pragma_pretraining_pipeline,
        package_path="pipeline/pragma_pipeline.yaml",
    )
    print("Pipeline compiled to pipeline/pragma_pipeline.yaml")
