"""PRAGMA workbench API — launch and inspect PRAGMA pretraining from a notebook.

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
    get_dspa_endpoint       — resolve KFP API endpoint URL from env/defaults
    get_service_account_token — read SA token from pod mount; never printed
    DSPAConfig              — lightweight endpoint + auth config value object
    make_kfp_client         — construct kfp.Client (lazy kfp import)
    upload_pipeline         — upload compiled YAML to DSPA
    submit_pipeline_run     — create a KFP pipeline run

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
ADR: docs/decisions/004-workbench-decorated-pipelines.md
"""

from src.workbench._api import train_pragma
from src.workbench._decorators import PragmaPipeline, pragma_pipeline
from src.workbench._intent import dataset, train
from src.workbench._run import (
    PIPELINE_STEP_NAMES,
    PipelineStep,
    PragmaRun,
    PragmaRunProtocol,
)
from src.workbench._submit import (
    DSPAConfig,
    get_dspa_endpoint,
    get_service_account_token,
    make_kfp_client,
    submit_pipeline_run,
    upload_pipeline,
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
]
