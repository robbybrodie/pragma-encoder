"""Platform-aware optional subpackage: PRAGMA workbench API.

This subpackage provides KFP pipeline authoring and DSPA submission helpers
for use inside OpenShift AI Workbench pods. It is **platform-aware by design**:

  - Reads Kubernetes service account tokens from pod-mounted paths
  - Constructs DSPA/KFP endpoint URLs from namespace information
  - Wraps kfp.Client for pipeline upload and run submission
  - Guards kfp-kubernetes as a lazy import for Kubernetes-native pipeline features

This subpackage is NOT part of the platform-neutral core wheel. It is installed
under the ``[workbench]`` optional extras in pyproject.toml only::

    pip install 'pragma-encoder[workbench]'

Importing the top-level ``pragma_encoder`` package does NOT import this
subpackage. It must be imported explicitly::

    from pragma_encoder.workbench import train_pragma

TD-009: the eventual goal is to move this subpackage to a separate distribution
package so the core wheel has zero platform dependencies. Until then it lives
here and is labelled as a platform-aware optional layer.
See docs/tech-debt.md — TD-009.

Public surface:
    train_pragma    — launch PRAGMA pretraining; mode="dry_run" for preview,
                      mode="pipeline" for decorator/compile path
    pragma_pipeline — @pragma_pipeline decorator for pipeline authoring DSL
    dataset         — intent-capture function for dataset selection
    train           — intent-capture function for training parameters
    PragmaPipeline  — result of @pragma_pipeline decoration
    PragmaRun       — result of a train_pragma() invocation
    PragmaRunProtocol, PipelineStep, PIPELINE_STEP_NAMES — supporting types

    DSPA/KFP v2 submit path (src/workbench/_submit.py):
    get_dspa_endpoint         — resolve KFP API endpoint URL from env/defaults
    get_service_account_token — read SA token from pod mount; never printed
    DSPAConfig                — lightweight endpoint + auth config value object
    make_kfp_client           — construct kfp.Client (lazy kfp import)
    upload_pipeline           — upload compiled YAML to DSPA
    submit_pipeline_run       — create a KFP pipeline run

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
ADR: docs/decisions/004-workbench-decorated-pipelines.md
"""

from pragma_encoder.workbench._api import train_pragma
from pragma_encoder.workbench._decorators import PragmaPipeline, pragma_pipeline
from pragma_encoder.workbench._intent import dataset, train
from pragma_encoder.workbench._run import (
    PIPELINE_STEP_NAMES,
    PipelineStep,
    PragmaRun,
    PragmaRunProtocol,
)
from pragma_encoder.workbench._submit import (
    DSPAConfig,
    get_dspa_endpoint,
    get_run_status,
    get_service_account_token,
    make_kfp_client,
    submit_pipeline_run,
    upload_pipeline,
    wait_for_run_terminal,
)

__all__ = [
    "train_pragma",
    "pragma_pipeline",
    "dataset",
    "train",
    "PragmaPipeline",
    "PragmaRun",
    "PragmaRunProtocol",
    "PipelineStep",
    "PIPELINE_STEP_NAMES",
    # DSPA/KFP v2 submit path
    "DSPAConfig",
    "get_dspa_endpoint",
    "get_service_account_token",
    "make_kfp_client",
    "upload_pipeline",
    "submit_pipeline_run",
    "get_run_status",
    "wait_for_run_terminal",
]
