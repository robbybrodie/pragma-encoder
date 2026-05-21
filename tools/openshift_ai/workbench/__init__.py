"""OpenShift AI workbench helper — platform-aware tooling, not part of the core wheel.

This package provides KFP pipeline authoring and DSPA submission helpers
for use inside OpenShift AI Workbench pods.  It is platform-aware by design:

  - Reads Kubernetes service account tokens from pod-mounted paths
  - Constructs DSPA/KFP endpoint URLs from namespace information
  - Wraps kfp.Client for pipeline upload and run submission
  - Guards kfp-kubernetes as a lazy import for Kubernetes-native pipeline features

**This package is NOT part of the pragma_encoder wheel.**

It lives under tools/openshift_ai/workbench/ in the repository, outside the
src/pragma_encoder/ package tree.  It depends on the pragma_encoder wheel but
pragma_encoder does not import or depend on this tooling.

Install requirements for this tooling::

    pip install kfp>=2 kfp-kubernetes>=1.2
    pip install pragma-encoder   # the core wheel

Usage (from a workbench notebook)::

    import sys
    sys.path.insert(0, "/path/to/pragma-encoder")   # repo root on sys.path

    from tools.openshift_ai.workbench import train_pragma

Public surface:
    train_pragma    — launch PRAGMA pretraining; mode="dry_run" for preview,
                      mode="pipeline" for decorator/compile path
    pragma_pipeline — @pragma_pipeline decorator for pipeline authoring DSL
    dataset         — intent-capture function for dataset selection
    train           — intent-capture function for training parameters
    PragmaPipeline  — result of @pragma_pipeline decoration
    PragmaRun       — result of a train_pragma() invocation
    PragmaRunProtocol, PipelineStep, PIPELINE_STEP_NAMES — supporting types

    DSPA/KFP v2 submit path (_submit.py):
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

from tools.openshift_ai.workbench._api import train_pragma
from tools.openshift_ai.workbench._decorators import PragmaPipeline, pragma_pipeline
from tools.openshift_ai.workbench._intent import dataset, train
from tools.openshift_ai.workbench._run import (
    PIPELINE_STEP_NAMES,
    PipelineStep,
    PragmaRun,
    PragmaRunProtocol,
)
from tools.openshift_ai.workbench._submit import (
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
