"""PRAGMA workbench API — launch and inspect PRAGMA pretraining from a notebook.

Public surface:
    train_pragma  — launch PRAGMA pretraining; mode="dry_run" for preview
    PragmaRun     — result of a train_pragma() invocation
    PragmaRunProtocol, PipelineStep, PIPELINE_STEP_NAMES — supporting types

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

from src.workbench._api import train_pragma
from src.workbench._run import (
    PragmaRun,
    PragmaRunProtocol,
    PipelineStep,
    PIPELINE_STEP_NAMES,
)

__all__ = [
    "train_pragma",
    "PragmaRun",
    "PragmaRunProtocol",
    "PipelineStep",
    "PIPELINE_STEP_NAMES",
]
