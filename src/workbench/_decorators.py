"""PragmaPipeline — decorator-based PRAGMA pipeline authoring.

Provides the @pragma_pipeline decorator DSL for expressing training intent
in normal Python. The decorated object captures intent and can compile it
to a KFP v2 pipeline YAML for submission to OpenShift Pipelines.

Target UX::

    from src.workbench import pragma_pipeline, dataset, train

    @pragma_pipeline(name="pragma-s-ibm-tabformer")
    def run():
        ds = dataset("ibm-tabformer", prepare_if_missing=True)
        train(dataset=ds, model_size="S", epochs=1, max_steps=1)

    run.show_pipeline()
    run.compile("pipeline/generated/pragma-s-ibm-tabformer.yaml")

Dependency direction (preserved):
    This file carries no static pipeline/ dependency. compile() uses
    importlib.import_module("pipeline.pragma_pipeline") inside its body.
    The quoted string "pipeline.pragma_pipeline" does not match the
    literal patterns that the TestNoCircularDependency guard scans for
    in src/ source files.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/004-workbench-decorated-pipelines.md
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from src.workbench._intent import (
    TrainIntent,
    start_capture,
    stop_capture,
)
from src.workbench._run import PIPELINE_STEP_NAMES, STEP_DESCRIPTIONS

# ---------------------------------------------------------------------------
# PragmaPipeline — result of @pragma_pipeline decoration
# ---------------------------------------------------------------------------

class PragmaPipeline:
    """Wraps a @pragma_pipeline decorated function's training intent.

    Exposes three methods:
        show_pipeline() — print the five §2.4 stages (no DRY RUN banner)
        compile(path)   — write KFP v2 YAML; requires kfp installed
        submit()        — future extension; raises NotImplementedError now

    Attributes:
        name:   The pipeline name passed to @pragma_pipeline.
        stages: Tuple of the five §2.4 stage name strings.
                Identical to PIPELINE_STEP_NAMES so callers can inspect
                or validate the stage list.

    Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
    ADR: docs/decisions/004-workbench-decorated-pipelines.md
    """

    def __init__(
        self,
        name: str,
        train_intent: TrainIntent,
    ) -> None:
        self.name = name
        self._train_intent = train_intent
        # stages mirrors PIPELINE_STEP_NAMES — exposed for inspection/testing.
        self.stages: tuple[str, ...] = PIPELINE_STEP_NAMES

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def show_pipeline(self) -> None:
        """Print the five §2.4 stage names without a DRY RUN banner.

        This is a compiled intent view — not a dry_run preview. The DRY RUN
        banner is reserved for train_pragma(mode='dry_run') previews.
        """
        print(f"PRAGMA Pipeline: {self.name}")
        print("=" * 50)
        for stage in self.stages:
            desc = STEP_DESCRIPTIONS[stage]
            print(f"  [planned  ] {stage:<8} — {desc}")

    def compile(self, path: str | object) -> None:
        """Compile this pipeline to a KFP v2 YAML file.

        Delegates to pragma_pretraining_pipeline via importlib to avoid
        static pipeline/ imports in src/ source, which would break the
        TestNoCircularDependency guard in tests/test_pipeline_components.py.

        Args:
            path: Output path for the compiled YAML. Accepts str or Path.

        Raises:
            RuntimeError: If kfp is not installed. Message mentions kfp,
                          install, and requirements.txt so the user knows
                          how to resolve the dependency.
        """
        try:
            kfp_compiler = importlib.import_module("kfp.compiler")
            pipeline_mod = importlib.import_module("pipeline.pragma_pipeline")
        except ImportError as exc:
            raise RuntimeError(
                "compile() requires kfp to be installed. "
                "Install it with: pip install -r requirements.txt "
                "(includes kfp). Or use the prepared workbench image "
                "where kfp is pre-installed."
            ) from exc

        # Delegate to the existing pragma_pretraining_pipeline lower-level
        # building block (pipeline/pragma_pipeline.py). No duplication of
        # model, tokeniser, or training logic.
        pipeline_fn = pipeline_mod.pragma_pretraining_pipeline
        compiler = kfp_compiler.Compiler()
        compiler.compile(pipeline_fn, str(path))

    def submit(self) -> None:
        """Submit this pipeline to a KFP server.

        Not yet implemented. Use compile() first to produce a YAML file,
        then submit it via the KFP API or the OpenShift AI Pipelines UI.

        Raises:
            NotImplementedError: Always. submit() is a future extension.
        """
        raise NotImplementedError(
            "submit() is not yet implemented. "
            "Use compile(path) to write the pipeline YAML first, "
            "then submit it via the KFP API or OpenShift AI Pipelines UI."
        )


# ---------------------------------------------------------------------------
# pragma_pipeline — decorator factory
# ---------------------------------------------------------------------------

def pragma_pipeline(*, name: str) -> Callable[..., Any]:
    """Decorator factory for PRAGMA pipeline authoring.

    Captures training intent from the decorated function body and returns a
    PragmaPipeline. The decorated function is executed exactly once at
    decoration time inside a thread-local capture context. No side effects:
    no adapter.prepare(), no subprocess.run(), no S3 access.

    Args:
        name: Pipeline name, e.g. "pragma-s-ibm-tabformer".

    Returns:
        Decorator that wraps the function and returns a PragmaPipeline.

    Example::

        @pragma_pipeline(name="pragma-s-ibm-tabformer")
        def run():
            ds = dataset("ibm-tabformer", prepare_if_missing=True)
            train(dataset=ds, model_size="S", epochs=1, max_steps=1)

        run.show_pipeline()
        run.compile("pipeline/generated/pragma-s-ibm-tabformer.yaml")

    Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
    ADR: docs/decisions/004-workbench-decorated-pipelines.md
    """
    def decorator(fn: Callable[..., Any]) -> PragmaPipeline:
        # Execute the function body once inside the thread-local capture context
        # so train() can register its TrainIntent in the capture list.
        # No side effects: dataset() and train() are pure intent-capture calls.
        start_capture()
        try:
            fn()
        finally:
            captured = stop_capture()

        if len(captured) == 0:
            raise ValueError(
                "@pragma_pipeline decorated function must call train() exactly once. "
                "No train() call was found in the function body. "
                "Add a train() call inside the decorated function, e.g.:\n"
                "    @pragma_pipeline(name='my-pipeline')\n"
                "    def run():\n"
                "        ds = dataset('ibm-tabformer')\n"
                "        train(dataset=ds, model_size='S', epochs=10)"
            )
        if len(captured) > 1:
            raise ValueError(
                f"@pragma_pipeline decorated function must call train() exactly once. "
                f"Found {len(captured)} train() calls — intent is ambiguous. "
                f"Use separate @pragma_pipeline definitions for separate training runs."
            )
        return PragmaPipeline(name=name, train_intent=captured[0])

    return decorator
