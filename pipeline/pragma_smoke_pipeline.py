"""Minimal KFP v2 smoke pipeline for Level 3 DSPA integration tests.

Implements a single self-contained KFP v2 component that runs:
  1. python -m pragma_encoder.data.fit_tokenizer -- fits tokenizer on inline synthetic CSV
  2. python -m pragma_encoder.training.train -- PRAGMA-S training with --max-steps 1

Purpose:
  Level 3 (DSPA/KFP v2 pipeline smoke) requires a component that can run
  inside a standard KFP component pod without S3, distributed training, or persistent volumes
  infrastructure. Production pipeline components (components_pragma.py)
  require all three -- they are not suitable for smoke testing.

  This module fills that gap: one component, one pod, no external dependencies.

Architecture boundary:
  This module is DIAGNOSTIC. It proves the KFP v2 pipeline path works.
  It does NOT replace the production pipeline (pragma_pipeline.py).
  Do not add production pipeline logic here.

  Level 3 PASS is independent of Level 3b PASS:
    Level 3b (batch/v1 Job) -> image is pullable and training scripts run
    Level 3  (KFP v2 Run)   -> DSPA pipeline orchestration works end-to-end

Image requirement:
  The component base_image must contain the pragma_encoder wheel and all
  Python dependencies. The PRAGMA training image is the correct choice
  (see openshift/training/Dockerfile.training).

  The image guarantees:
    - python -m pragma_encoder.data.fit_tokenizer  works (wheel installed)
    - python -m pragma_encoder.training.train  works (wheel installed; replaces scripts/train_pragma.py)

  The workbench image must NOT be used here -- it does not install the
  pragma_encoder wheel. Component pods would fail with:
    ModuleNotFoundError: No module named 'pragma_encoder'

  Override at compile time via env var (read once at module import):
    PRAGMA_TRAINING_IMAGE      -- training image URI (highest priority)
    PRAGMA_KFP_COMPONENT_IMAGE -- explicit KFP component image override

KFP optional import guard:
  kfp is an optional dependency (pyproject.toml [workbench] extra).
  This module guards the kfp import exactly as components_pragma.py does:
  try/except at module level; identity decorator fallback.

Reference: Level 3 -- OpenShift AI KFP v2 pipeline smoke
Test: tests/openshift/test_03_pipeline_smoke_run.py
Units: tests/test_smoke_pipeline.py
"""

import os

# ---------------------------------------------------------------------------
# KFP optional import guard -- mirrors components_pragma.py pattern
# ---------------------------------------------------------------------------

try:
    from kfp import dsl as _dsl
    _KFP_AVAILABLE: bool = True
except ImportError:
    _dsl = None  # type: ignore[assignment]
    _KFP_AVAILABLE: bool = False


def _component(**kwargs):  # type: ignore[no-untyped-def]
    """Return @dsl.component decorator if KFP available; identity decorator otherwise.

    Sets ``__wrapped__`` on the decorated function so ``inspect.signature()``
    follows the original declared parameters, not the kfp runtime wrapper.
    """
    if _KFP_AVAILABLE:
        kfp_deco = _dsl.component(**kwargs)
        def _wrap(fn):  # type: ignore[no-untyped-def]
            decorated = kfp_deco(fn)
            decorated.__wrapped__ = fn  # preserve original signature for inspect
            return decorated
        return _wrap
    return lambda fn: fn


def _pipeline(**kwargs):  # type: ignore[no-untyped-def]
    """Return @dsl.pipeline decorator if KFP available; identity decorator otherwise.

    Sets ``__wrapped__`` on the decorated function so ``inspect.signature()``
    follows the original declared parameters, not the kfp runtime wrapper.
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
# Component base image
#
# Read at import time -- KFP @dsl.component captures base_image at decoration
# time, so the env var must be set before this module is imported (i.e., at
# compile time in the workbench).
#
# Priority:
#   1. PRAGMA_KFP_COMPONENT_IMAGE  (explicit override)
#   2. PRAGMA_TRAINING_IMAGE       (training image, preferred)
#   3. Cluster-internal default    (hardcoded fallback)
#
# The workbench image (pragma-encoder-workbench) must NOT be used here --
# it is a deps-only image without PRAGMA source code. Component pods would
# fail with ModuleNotFoundError: No module named 'src'.
# ---------------------------------------------------------------------------

_DEFAULT_SMOKE_IMAGE = (
    "image-registry.openshift-image-registry.svc:5000"
    "/pragma-encoder/pragma-encoder-training:latest"
)

_SMOKE_IMAGE: str = (
    os.environ.get("PRAGMA_KFP_COMPONENT_IMAGE")
    or os.environ.get("PRAGMA_TRAINING_IMAGE")
    or _DEFAULT_SMOKE_IMAGE
)


# ---------------------------------------------------------------------------
# Smoke training component
# ---------------------------------------------------------------------------

@_component(base_image=_SMOKE_IMAGE)
def pragma_smoke_training(max_steps: int = 1, output_dir: str = "/tmp/pragma-smoke-output") -> None:
    """Run PRAGMA-S smoke training inside a single KFP component pod.

    Steps:
      1. Write a 30-row synthetic TabFormer CSV to /tmp/pragma-smoke/.
      2. Run python -m pragma_encoder.data.fit_tokenizer to build vocab.pkl.
      3. Run python -m pragma_encoder.training.train --max-steps N --model-variant pragma-s.
      4. Print artifact location (checkpoint files, metadata.json presence).
      5. Print completion marker: 'PRAGMA smoke training completed'.

    No S3, no distributed training, no persistent volumes required.
    Input data is written to /tmp/pragma-smoke/ (ephemeral).
    Training artifacts (checkpoint, metadata.json) are written to output_dir.
    The component pod exits 0 on success.

    Args:
        max_steps:  Number of training steps before early stop. Default: 1.
        output_dir: Directory for checkpoint and metadata.json artifacts.
                    Default: /tmp/pragma-smoke-output (ephemeral pod storage).

    Log markers asserted by test_03_pipeline_smoke_run.py:
      'PRAGMA-S'                        -- printed by pragma_encoder.training.train
      'Reached --max-steps'             -- printed by pragma_encoder.training.train early-stop
      'PRAGMA smoke training completed' -- printed by this component
    """
    # All imports inside the function body -- KFP serialises this function
    # and runs it in a fresh Python process inside the component pod.
    import pathlib
    import subprocess
    import sys
    import textwrap

    # ------------------------------------------------------------------
    # Step 1 -- No script location needed.
    #
    # Training is invoked as a module (python -m pragma_encoder.training.train)
    # because the pragma_encoder wheel is installed in site-packages -- no src/ tree
    # and no scripts/ directory search required.
    # fit_tokenizer is similarly invoked as a module.
    # ------------------------------------------------------------------
    print("[smoke] pragma_encoder.training.train invoked as a module (wheel-based image)")

    # ------------------------------------------------------------------
    # Step 2 -- Write inline TabFormer CSV.
    #
    # Same 30-row synthetic dataset as Level 3b (test_03b_training_job_smoke.py).
    # 10 users x 3 transactions. fit_tokenizer.py splits 80/20 by User ID:
    #   users 0-7 -> train (8 users), users 8-9 -> val (2 users).
    #
    # fit_tokenizer.py reads: data/tabformer/card_transaction.v1.csv (relative)
    # fit_tokenizer.py writes: data/tabformer/vocab.pkl (relative)
    # Both paths are relative to the CWD at runtime -- we pass cwd=work_dir.
    # ------------------------------------------------------------------
    work_dir = pathlib.Path("/tmp/pragma-smoke")
    data_dir = work_dir / "data" / "tabformer"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_content = textwrap.dedent("""\
        User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,MCC,Errors?,Is Fraud?
        0,0,2023,1,5,09:00,$12.50,Swipe Transaction,Coffee House,Sydney,NSW,5812,,No
        0,0,2023,1,6,12:30,$45.00,Chip Transaction,Grocery World,Melbourne,VIC,5411,,No
        0,0,2023,1,7,18:00,$8.75,Swipe Transaction,Fast Bites,Brisbane,QLD,5812,,No
        1,0,2023,1,5,10:00,$23.00,Swipe Transaction,Fuel Stop,Perth,WA,5541,,No
        1,0,2023,1,6,14:00,$67.50,Chip Transaction,Supermart,Adelaide,SA,5411,,No
        1,0,2023,1,7,19:00,$15.00,Swipe Transaction,Pizza Place,Hobart,TAS,5812,,No
        2,0,2023,1,5,08:30,$9.50,Swipe Transaction,Bakery Lane,Sydney,NSW,5461,,No
        2,0,2023,1,6,11:00,$34.00,Chip Transaction,Dept Store,Melbourne,VIC,5311,,No
        2,0,2023,1,7,17:30,$5.25,Swipe Transaction,Snack Bar,Brisbane,QLD,5812,,No
        3,0,2023,1,5,09:45,$78.00,Chip Transaction,Electronics Co,Perth,WA,5734,,No
        3,0,2023,1,6,13:30,$22.00,Swipe Transaction,Bookshop,Adelaide,SA,5942,,No
        3,0,2023,1,7,20:00,$11.50,Swipe Transaction,Cafe Nero,Hobart,TAS,5812,,No
        4,0,2023,1,5,07:00,$5.00,Swipe Transaction,Morning Brew,Sydney,NSW,5812,,No
        4,0,2023,1,6,10:30,$150.00,Chip Transaction,Fashion Store,Melbourne,VIC,5621,,No
        4,0,2023,1,7,16:00,$28.75,Swipe Transaction,Thai Kitchen,Brisbane,QLD,5812,,No
        5,0,2023,1,5,11:00,$44.00,Chip Transaction,Hardware Plus,Perth,WA,5251,,No
        5,0,2023,1,6,15:00,$18.50,Swipe Transaction,Juice Bar,Adelaide,SA,5812,,No
        5,0,2023,1,7,21:00,$92.00,Chip Transaction,Sports Gear,Hobart,TAS,5941,,No
        6,0,2023,1,5,08:00,$7.50,Swipe Transaction,News Stand,Sydney,NSW,5994,,No
        6,0,2023,1,6,12:00,$55.00,Chip Transaction,Pharmacy,Melbourne,VIC,5912,,No
        6,0,2023,1,7,18:30,$33.00,Swipe Transaction,Italian Rest,Brisbane,QLD,5812,,No
        7,0,2023,1,5,10:15,$19.00,Swipe Transaction,Florist,Perth,WA,5992,,No
        7,0,2023,1,6,14:30,$62.00,Chip Transaction,Furniture Co,Adelaide,SA,5712,,No
        7,0,2023,1,7,19:30,$14.25,Swipe Transaction,Taco Truck,Hobart,TAS,5812,,No
        8,0,2023,1,5,09:30,$38.00,Chip Transaction,Bike Shop,Sydney,NSW,5941,,No
        8,0,2023,1,6,13:00,$8.00,Swipe Transaction,Hot Dog Stand,Melbourne,VIC,5812,,No
        8,0,2023,1,7,17:00,$125.00,Chip Transaction,Jewellery Box,Brisbane,QLD,5944,,No
        9,0,2023,1,5,07:30,$6.50,Swipe Transaction,Milk Bar,Perth,WA,5812,,No
        9,0,2023,1,6,11:30,$41.00,Chip Transaction,Auto Parts,Adelaide,SA,5533,,No
        9,0,2023,1,7,20:30,$16.75,Swipe Transaction,Sushi Bar,Hobart,TAS,5812,,No
    """)

    csv_path = data_dir / "card_transaction.v1.csv"
    # Strip leading whitespace from textwrap.dedent result (indented block).
    csv_path.write_text("\n".join(line.strip() for line in csv_content.splitlines()))
    print(f"[smoke] CSV written: {csv_path} ({len(csv_path.read_text().splitlines())} lines)")

    vocab_path = work_dir / "data" / "tabformer" / "vocab.pkl"

    # ------------------------------------------------------------------
    # Step 3 -- Fit tokenizer via module invocation.
    #
    # pragma_encoder.data.fit_tokenizer is installed in site-packages by the
    # wheel. It uses hardcoded relative paths for read/write (CSV_PATH and
    # VOCAB_PATH relative to CWD), so we pass cwd=work_dir.
    # ------------------------------------------------------------------
    print("[smoke] Running python -m pragma_encoder.data.fit_tokenizer ...")
    subprocess.run(
        [sys.executable, "-m", "pragma_encoder.data.fit_tokenizer"],
        cwd=str(work_dir),
        check=True,
    )

    if not vocab_path.exists():
        raise RuntimeError(
            f"fit_tokenizer.py did not create {vocab_path}. "
            "Check fit_tokenizer.py output above for errors."
        )
    print(f"[smoke] vocab.pkl created: {vocab_path}")

    # ------------------------------------------------------------------
    # Step 4 -- Run PRAGMA-S training with --max-steps.
    #
    # Invoked as a module (python -m pragma_encoder.training.train) -- no
    # file-path search needed; the wheel installs the module into site-packages.
    # Uses the same argument set as the Level 3b smoke shell to ensure
    # training produces the expected log markers:
    #   'PRAGMA-S'             -- model variant confirmation
    #   'Reached --max-steps'  -- early-stop marker
    # ------------------------------------------------------------------
    _output_dir = pathlib.Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] Running pragma_encoder.training.train --max-steps {max_steps} --model-variant pragma-s ...")
    subprocess.run(
        [
            sys.executable, "-m", "pragma_encoder.training.train",
            "--csv-path", str(csv_path),
            "--vocab-path", str(vocab_path),
            "--output-dir", str(_output_dir),
            "--model-variant", "pragma-s",
            "--epochs", "1",
            "--num-workers", "0",
            "--batch-size", "1",
            "--max-steps", str(max_steps),
            "--device", "cpu",
        ],
        check=True,
    )

    # ------------------------------------------------------------------
    # Step 5 -- Print artifact location and completion marker.
    # ------------------------------------------------------------------
    _ckpt_files = sorted(_output_dir.glob("checkpoint_epoch*.pt"))
    _meta = _output_dir / "metadata.json"
    print(f"[smoke] Training artifacts written to: {_output_dir}")
    print(f"[smoke] Checkpoints found: {[f.name for f in _ckpt_files]}")
    print(f"[smoke] metadata.json present: {_meta.exists()}")
    print("PRAGMA smoke training completed")


# ---------------------------------------------------------------------------
# Smoke pipeline
# ---------------------------------------------------------------------------

@_pipeline(
    name="pragma-smoke-training-pipeline",
    description=(
        "Minimal PRAGMA-S smoke pipeline for Level 3 DSPA/KFP v2 integration testing. "
        "Runs fit_tokenizer + train_pragma --max-steps N in a single component pod. "
        "No S3, no distributed training, no persistent volumes -- all data is ephemeral (/tmp)."
    ),
)
def pragma_smoke_training_pipeline(
    max_steps: int = 1,
    output_dir: str = "/tmp/pragma-smoke-output",
) -> None:
    """Single-component KFP v2 pipeline for Level 3 DSPA smoke testing.

    Compiles to a KFP v2 YAML suitable for upload to the DSPA KFP v2 API.
    The pipeline has one component (pragma_smoke_training) that runs entirely
    inside a single pod without external infrastructure dependencies.

    Args:
        max_steps:  Number of training steps before early stop. Default 1.
                    The test uses max_steps=1 to minimise pod runtime.
        output_dir: Directory inside the pod for checkpoint and metadata.json.
                    Default: /tmp/pragma-smoke-output (ephemeral pod storage).
                    Override to a mounted volume path to inspect artifacts after the run.
    """
    pragma_smoke_training(max_steps=max_steps, output_dir=output_dir)
