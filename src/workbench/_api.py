"""train_pragma() — workbench entry point for PRAGMA pretraining.

Maps the five §2.4 pipeline stages onto a concrete execution path and
returns a PragmaRun (or PragmaPipeline for mode="pipeline") for inspection.

Supported modes:
    "dry_run"  — return a PragmaRun immediately without doing any work.
                 Constructs a placeholder DatasetManifest from the requested
                 dataset name and PRAGMAConfig. No S3, no cluster, no subprocess.
                 Useful for previewing pipeline shape and configuration.
    "pipeline" — return a PragmaPipeline capturing training intent.
                 No training, no S3, no cluster at call time.
                 Call .compile(path) to produce a KFP v2 YAML.
    "auto"     — detect environment: OpenShift pod → "cluster"; else → "local".
    "cluster"  — submit a KFTO PyTorchJob via oc apply (not yet implemented).
    "local"    — run scripts/train_pragma.py as a subprocess.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
ADR: docs/decisions/004-workbench-decorated-pipelines.md
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workbench._decorators import PragmaPipeline

from src.data.adapters import get_adapter
from src.model.config import PRAGMAConfig
from src.workbench._run import (
    STEP_DESCRIPTIONS,
    PipelineStep,
    PragmaRun,
)

# ---------------------------------------------------------------------------
# Model size → PRAGMAConfig factory map (case-sensitive per §2.4)
# ---------------------------------------------------------------------------

_MODEL_SIZE_MAP: dict[str, type] = {
    "S": PRAGMAConfig.pragma_s,
    "M": PRAGMAConfig.pragma_m,
    "L": PRAGMAConfig.pragma_l,
}

# Model size → scripts/train_pragma.py --model-variant value
_MODEL_VARIANT_MAP: dict[str, str] = {
    "S": "pragma-s",
    "M": "pragma-m",
    "L": "pragma-l",
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
    local_csv_path: str | None = None,
    output_dir: str | None = None,
    max_steps: int | None = None,
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
        local_csv_path:     Path to source CSV file. Required for mode="local".
        output_dir:         Directory to write checkpoints and vocab. Used in
                            mode="local". Defaults to outputs/pragma-local/<timestamp>.
        max_steps:          Maximum training steps. Passed as --max-steps to
                            scripts/train_pragma.py. Useful for smoke tests.

    Returns:
        PragmaRun: wraps the submitted job. Call show_pipeline(), metrics(),
                   artifacts() to inspect progress and results.

    Raises:
        ValueError:         If model_size is not "S", "M", or "L" (case-sensitive).
        KeyError:           If dataset is not registered in the adapter registry.
        NotImplementedError: If mode="local" and nodes > 1.
        FileNotFoundError:  If prepare_if_missing=False and dataset data is missing.
    """
    # --- Validate model_size (case-sensitive) before any other work ---
    if model_size not in _MODEL_SIZE_MAP:
        raise ValueError(
            f"model_size must be one of 'S', 'M', 'L' (case-sensitive), "
            f"got {model_size!r}. "
            f"Valid values: {sorted(_MODEL_SIZE_MAP)}"
        )

    # --- Validate dataset via adapter registry (raises KeyError if unknown) ---
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

    if effective_mode == "pipeline":
        return _build_pipeline_intent(
            dataset=dataset,
            model_size=model_size,
            epochs=epochs,
            max_steps=max_steps,
        )

    if effective_mode == "local":
        return _build_local_run(
            dataset=dataset,
            model_size=model_size,
            config=config,
            local_csv_path=local_csv_path,
            output_dir=output_dir,
            epochs=epochs,
            nodes=nodes,
            max_steps=max_steps,
        )

    raise NotImplementedError(
        f"mode='{effective_mode}' is not yet implemented. "
        f"Use mode='dry_run' to preview the pipeline shape and configuration, "
        f"or mode='local' to run training as a subprocess."
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


# ---------------------------------------------------------------------------
# Private — pipeline mode builder
# ---------------------------------------------------------------------------

def _build_pipeline_intent(
    dataset: str,
    model_size: str,
    epochs: int,
    max_steps: int | None,
) -> PragmaPipeline:
    """Return a PragmaPipeline capturing training intent without executing anything.

    No training, no S3, no cluster access. The returned PragmaPipeline can be
    inspected via show_pipeline() and compiled to KFP YAML via compile(path).

    Args:
        dataset:    Dataset registry key, e.g. "ibm-tabformer".
        model_size: Model size variant: "S", "M", or "L".
        epochs:     Number of pretraining epochs.
        max_steps:  Maximum training steps, or None.

    Returns:
        PragmaPipeline wrapping the captured training intent.
    """
    from src.workbench._decorators import PragmaPipeline
    from src.workbench._intent import DatasetIntent, TrainIntent

    pipeline_name = f"pragma-{model_size.lower()}-{dataset}"
    ds_intent = DatasetIntent(name=dataset, prepare_if_missing=True)
    train_intent = TrainIntent(
        dataset=ds_intent,
        model_size=model_size,
        epochs=epochs,
        max_steps=max_steps,
    )
    return PragmaPipeline(name=pipeline_name, train_intent=train_intent)


# ---------------------------------------------------------------------------
# Private — local mode builder
# ---------------------------------------------------------------------------

def _build_local_run(
    dataset: str,
    model_size: str,
    config: PRAGMAConfig,
    local_csv_path: str | None,
    output_dir: str | None,
    epochs: int,
    nodes: int,
    max_steps: int | None,
) -> PragmaRun:
    """Run scripts/train_pragma.py as a local subprocess and return PragmaRun.

    Stage statuses:
        prepare  — completed (adapter.prepare() is called with upload=False)
        upload   — skipped   (no S3 in local mode)
        submit   — skipped   (no KFTO cluster in local mode)
        train    — completed | failed (based on subprocess returncode)
        export   — skipped   (no S3 export in local mode)

    No S3, no oc, no kubectl, no kfp. Single-node only (nodes=1).

    Args:
        dataset:        Dataset registry key.
        model_size:     Model size variant "S", "M", or "L".
        config:         PRAGMAConfig built from model_size.
        local_csv_path: Path to source CSV file. Required.
        output_dir:     Directory to write checkpoints and vocab.
        epochs:         Number of pretraining epochs.
        nodes:          Number of training nodes (must be 1).
        max_steps:      Maximum training steps (passed to train script), or None.

    Returns:
        PragmaRun with run_mode="local" and artifacts={"output_dir": ...}.

    Raises:
        NotImplementedError: If nodes > 1.
        ValueError:          If local_csv_path is None.
    """
    # --- Validate: local mode is single-node only ---
    if nodes > 1:
        raise NotImplementedError(
            f"mode='local' supports nodes=1 only. "
            f"nodes={nodes} requires KFTO PyTorchJob orchestration (mode='cluster'). "
            f"See openshift/training/pytorchjob-pragma-s-2node.yaml for multi-node manifests."
        )

    # --- Validate: local_csv_path is required ---
    if local_csv_path is None:
        raise ValueError(
            "mode='local' requires local_csv_path. "
            "Provide the path to the source CSV file, e.g. "
            "train_pragma(mode='local', local_csv_path='data/card_transaction.v1.csv')."
        )

    # --- Resolve output_dir ---
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/pragma-local/{timestamp}"

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir_path / "vocab.pkl"

    # --- Stage 1: prepare — call adapter with upload=False (no S3) ---
    adapter_cls = get_adapter(dataset)
    adapter = adapter_cls(csv_path=local_csv_path, vocab_path=vocab_path)
    manifest = adapter.prepare(config, upload=False)

    # --- Build initial steps list ---
    steps = [
        PipelineStep("prepare", STEP_DESCRIPTIONS["prepare"], "completed"),
        PipelineStep("upload",  STEP_DESCRIPTIONS["upload"],  "skipped"),
        PipelineStep("submit",  STEP_DESCRIPTIONS["submit"],  "skipped"),
        PipelineStep("train",   STEP_DESCRIPTIONS["train"],   "pending"),
        PipelineStep("export",  STEP_DESCRIPTIONS["export"],  "skipped"),
    ]

    # --- Stage 4: train — invoke scripts/train_pragma.py as subprocess ---
    model_variant = _MODEL_VARIANT_MAP[model_size]
    cmd = [
        sys.executable, "scripts/train_pragma.py",
        "--csv-path",      str(local_csv_path),
        "--vocab-path",    str(vocab_path),
        "--output-dir",    str(output_dir_path),
        "--model-variant", model_variant,
        "--epochs",        str(epochs),
        "--num-workers",   "0",
        "--batch-size",    "1",
    ]
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])

    result = subprocess.run(cmd, check=False)
    train_status = "completed" if result.returncode == 0 else "failed"
    steps[3] = PipelineStep("train", STEP_DESCRIPTIONS["train"], train_status)

    return PragmaRun(
        manifest=manifest,
        steps=steps,
        run_mode="local",
        _artifacts={"output_dir": str(output_dir_path)},
    )
