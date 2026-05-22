"""Intent value objects for the PRAGMA workbench decorator DSL.

DatasetIntent and TrainIntent capture the data scientist's training intent
from the body of a @pragma_pipeline decorated function. They are pure value
objects with no side effects: no adapter lookup, no S3 access, no subprocess.

The thread-local capture context (start_capture / stop_capture /
_register_train_intent) is used by pragma_pipeline() in _decorators.py to
collect the TrainIntent expressed inside the decorated function body.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/004-workbench-decorated-pipelines.md
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Thread-local capture context (internal)
# ---------------------------------------------------------------------------

_capture = threading.local()


def _is_capturing() -> bool:
    """Return True if we are currently inside a @pragma_pipeline capture context."""
    return getattr(_capture, "active", False)


def start_capture() -> None:
    """Activate the thread-local intent capture context."""
    _capture.active = True
    _capture.intents = []


def stop_capture() -> list[TrainIntent]:
    """Deactivate the capture context and return all captured TrainIntents."""
    _capture.active = False
    intents = list(getattr(_capture, "intents", []))
    _capture.intents = []
    return intents


def _register_train_intent(intent: TrainIntent) -> None:
    """Append intent to the active capture context (no-op if not capturing)."""
    if _is_capturing():
        _capture.intents.append(intent)


# ---------------------------------------------------------------------------
# DatasetIntent — value object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetIntent:
    """Captures the data scientist's dataset choice.

    Returned by dataset(). Records the dataset name and whether to prepare
    the dataset if it is not yet in S3. Has no side effects at construction.

    Args:
        name:               Dataset registry key, e.g. "ibm-tabformer".
        prepare_if_missing: If True, run the dataset adapter if the dataset
                            is not already in S3. Default: True.

    Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
    ADR: docs/decisions/004-workbench-decorated-pipelines.md
    """

    name: str
    prepare_if_missing: bool = True


# ---------------------------------------------------------------------------
# TrainIntent — value object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainIntent:
    """Captures the data scientist's training intent.

    Returned by train(). Records the training parameters. Has no side effects.
    When called inside a @pragma_pipeline decorated function, the intent is
    also registered in the active thread-local capture context so PragmaPipeline
    can compile it later.

    Args:
        dataset:    The DatasetIntent expressing which dataset to train on.
        model_size: PRAGMA model size: "S", "M", or "L" (case-sensitive).
        epochs:     Number of pretraining epochs. Default: 10.
        max_steps:  Maximum training steps, or None for no limit. Default: None.

    Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
    ADR: docs/decisions/004-workbench-decorated-pipelines.md
    """

    dataset: DatasetIntent
    model_size: str
    epochs: int = 10
    max_steps: int | None = None


# ---------------------------------------------------------------------------
# Public intent-capture functions
# ---------------------------------------------------------------------------

def dataset(name: str, *, prepare_if_missing: bool = True) -> DatasetIntent:
    """Record dataset intent — no side effects.

    Returns a DatasetIntent value object. Does not look up any adapter,
    read any data, or access S3. Safe to call in any context.

    Args:
        name:               Dataset registry key, e.g. "ibm-tabformer".
        prepare_if_missing: If True, prepare the dataset if not yet in S3.
                            Default: True.

    Returns:
        DatasetIntent recording the name and prepare_if_missing flag.

    Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
    ADR: docs/decisions/004-workbench-decorated-pipelines.md
    """
    return DatasetIntent(name=name, prepare_if_missing=prepare_if_missing)


def train(
    *,
    dataset: DatasetIntent,
    model_size: str,
    epochs: int = 10,
    max_steps: int | None = None,
) -> TrainIntent:
    """Record training intent — no side effects.

    Returns a TrainIntent value object. Does not start any subprocess, access
    S3, or interact with any cluster resource. When called inside a
    @pragma_pipeline decorated function, the intent is registered in the
    active capture context so PragmaPipeline can compile it later.

    Args:
        dataset:    DatasetIntent returned by dataset().
        model_size: PRAGMA model size: "S", "M", or "L" (case-sensitive).
        epochs:     Number of pretraining epochs. Default: 10.
        max_steps:  Maximum training steps, or None for no limit. Default: None.

    Returns:
        TrainIntent recording the training parameters.

    Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
    ADR: docs/decisions/004-workbench-decorated-pipelines.md
    """
    intent = TrainIntent(
        dataset=dataset,
        model_size=model_size,
        epochs=epochs,
        max_steps=max_steps,
    )
    _register_train_intent(intent)
    return intent
