"""KFP SDK v2 pipeline definitions for PRAGMA pretraining.

Defines two PRAGMA pretraining pipelines for deployment on OpenShift AI
via Kubeflow Pipelines (KFP SDK v2).

pragma_pretraining_pipeline
    All five sec.2.4 stages always run: prepare -> upload -> submit -> train -> export.
    Accepts dataset_name (adapter registry key) and model_size.

pragma_train_from_manifest_pipeline
    Stages 3-5 only: submit -> train -> export.
    Accepts manifest_uri (existing DatasetManifest in S3) and model_size.
    Use when the dataset has already been prepared and uploaded.

The five pipeline stages mirror the sec.2.4 training infrastructure stages and
the PIPELINE_STEP_NAMES exposed by PragmaRun.show_pipeline():

    1. prepare   - prepare_dataset: fit tokeniser via DatasetAdapter
    2. upload    - upload_artifacts: upload prepared data to S3 (idempotent)
    3. submit    - submit_pytorchjob: configure / submit KFTO PyTorchJob
    4. train     - run_pretraining: execute pretraining (MEM objective sec.2.3.5)
    5. export    - export_checkpoint: upload model checkpoint to S3

Canonical dataset contract: DatasetManifest / manifest_uri (sec.2.4 data
storage).  No PVC-backed dataset storage.  All data lives in S3.

KFP is an optional dependency (same guard as components_pragma.py).
When kfp is not installed, @dsl.pipeline is not applied and the functions
are plain Python callables inspectable without a running KFP server.

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
    """Return @dsl.pipeline decorator if KFP available; identity decorator otherwise.

    When KFP is available, sets ``__wrapped__`` on the decorated function so that
    ``inspect.signature()`` follows through to the original function's declared
    parameters instead of the kfp runtime wrapper's ``(*args, **kwargs)`` signature.
    """
    if _KFP_AVAILABLE:
        kfp_deco = _dsl.pipeline(**kwargs)
        def _wrap(fn):  # type: ignore[no-untyped-def]
            decorated = kfp_deco(fn)
            decorated.__wrapped__ = fn  # preserve original signature for inspect
            return decorated
        return _wrap
    return lambda fn: fn


# ---------------------------------------------------------------------------
# Import pipeline components (pipeline/ -> src/ only; no circular dependency)
# ---------------------------------------------------------------------------

from pipeline.components_pragma import (  # noqa: E402, I001
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
        "Five sec.2.4 stages: prepare -> upload -> submit -> train -> export. "
        "All five stages always run. Use pragma_train_from_manifest_pipeline "
        "to start from an existing DatasetManifest URI (stages 3-5 only). "
        "Canonical dataset contract: DatasetManifest / manifest_uri (not PVC). "
        "Ostroukhov et al. (2026), arXiv:2604.08649v1."
    ),
)
def pragma_pretraining_pipeline(
    dataset_name: str,
    model_size: str = "S",
    epochs: int = 10,
    nodes: int = 1,
    max_steps: int = 0,
    limit_rows: int = 0,
) -> None:
    """Full PRAGMA pretraining pipeline: all five sec.2.4 stages always run.

    Runs prepare -> upload -> submit -> train -> export unconditionally.
    All five stages execute every time this pipeline is invoked.

    To resume training from an already-prepared DatasetManifest URI (stages
    3-5 only), use pragma_train_from_manifest_pipeline instead.

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
        max_steps:    Stop after this many training steps (0 = train all epochs).
                      Use for smoke/tiny runs without changing epochs.
        limit_rows:   Cap the training dataset to this many customers
                      (0 = use all). Use for smoke/tiny runs.
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
    submit_pytorchjob(
        manifest_uri=upload_op.output,
        model_size=model_size,
        nodes=nodes,
        epochs=epochs,
    )

    # Stage 4: Execute pretraining - masked event modelling (sec.2.3.5)
    train_op = run_pretraining(
        manifest_uri=upload_op.output,
        model_size=model_size,
        nodes=nodes,
        epochs=epochs,
        max_steps=max_steps,
        limit_rows=limit_rows,
    )

    # Stage 5: Export model checkpoints and outputs to S3
    export_checkpoint(
        checkpoint_uri=train_op.output,
        model_size=model_size,
    )


@_pipeline(
    name="pragma-train-from-manifest-pipeline",
    description=(
        "PRAGMA pretraining pipeline starting from an existing DatasetManifest URI. "
        "Runs sec.2.4 stages 3-5 only: submit -> train -> export. "
        "Stages 1 (prepare) and 2 (upload) are not run - the manifest already "
        "exists in S3. Use pragma_pretraining_pipeline to run all five stages. "
        "Ostroukhov et al. (2026), arXiv:2604.08649v1."
    ),
)
def pragma_train_from_manifest_pipeline(
    manifest_uri: str,
    model_size: str = "S",
    epochs: int = 10,
    nodes: int = 1,
) -> None:
    """PRAGMA pretraining pipeline from an existing DatasetManifest (stages 3-5).

    Accepts a pre-prepared DatasetManifest URI and runs only the cluster-facing
    stages: submit -> train -> export.  Stages 1 (prepare) and 2 (upload) are
    not run - the dataset is already prepared in S3.

    Use this pipeline to resume training or rerun pretraining on an existing
    prepared dataset without repeating the expensive tokeniser-fitting step.

    Args:
        manifest_uri: S3 URI of a prepared DatasetManifest.  Required - no default.
                      Canonical sec.2.4 training data contract.
        model_size:   Model variant "S", "M", or "L" (case-sensitive).
                      Maps to PRAGMAConfig.pragma_s/m/l() (Table 1).
                      Default: "S" (PRAGMA-S, ~10M parameters).
        epochs:       Number of pretraining epochs.  Default: 10.
        nodes:        Number of training nodes.
                      1 = single-node (pytorchjob-pragma-s.yaml),
                      2 = two-node DDP (pytorchjob-pragma-s-2node.yaml).
                      Default: 1.
    """
    # Stage 3: Configure and submit KFTO PyTorchJob
    submit_pytorchjob(
        manifest_uri=manifest_uri,
        model_size=model_size,
        nodes=nodes,
        epochs=epochs,
    )

    # Stage 4: Execute pretraining - masked event modelling (sec.2.3.5)
    train_op = run_pretraining(
        manifest_uri=manifest_uri,
        model_size=model_size,
        nodes=nodes,
        epochs=epochs,
    )

    # Stage 5: Export model checkpoints and outputs to S3
    export_checkpoint(
        checkpoint_uri=train_op.output,
        model_size=model_size,
    )


# ---------------------------------------------------------------------------
# Compilation entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _KFP_AVAILABLE:
        print("kfp is not installed - pipeline compilation requires kfp.")
        raise SystemExit(1)
    import kfp.compiler as compiler
    compiler.Compiler().compile(
        pipeline_func=pragma_pretraining_pipeline,
        package_path="pipeline/pragma_pipeline.yaml",
    )
    print("Pipeline compiled to pipeline/pragma_pipeline.yaml")
