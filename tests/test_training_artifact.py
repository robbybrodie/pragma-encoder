"""Tests for the training artifact contract: checkpoint + metadata.json.

Verifies that:
  - pragma_encoder.training.train produces a checkpoint and metadata.json
  - --limit-rows correctly caps the training dataset
  - pragma_pretraining_pipeline has max_steps / limit_rows parameters
  - pragma_smoke_training_pipeline has output_dir parameter
  - run_pretraining component uses wheel-based module invocation (not source-tree paths)
  - run_pretraining has scratch_dir parameter (configurable, not hardcoded /tmp/pragma-run)
  - run_pretraining wires --dataset-name and uploads metadata.json to S3
  - metadata.json dataset section uses csv_staging_path (not csv_path) and dataset_name
  - metadata.json args section is filtered by _METADATA_ALLOWED_ARG_KEYS (no credentials)
  - _safe_args_for_metadata excludes csv_path, vocab_path, device, and other staging paths

Training integration tests use a 30-row synthetic TabFormer CSV (same as the
Level 3 smoke component) with --limit-rows 2 --epochs 1 --device cpu.
All tests run locally without S3, distributed training, or cluster access.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
"""

from __future__ import annotations

import inspect
import json
import pathlib
import subprocess
import sys
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Inline CSV — 30 rows, 10 users × 3 transactions (same as smoke component)
# ---------------------------------------------------------------------------

_SMOKE_CSV = textwrap.dedent("""\
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


# ---------------------------------------------------------------------------
# Session-scoped fixture — runs training ONCE; all artifact tests share it
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def training_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run a tiny local training and return paths to the produced artifacts.

    Runs only once per test session (session scope) to keep the suite fast.
    Uses --limit-rows 2 --epochs 1 --device cpu (no --max-steps) so the epoch
    completes and a checkpoint is saved.  Finishes in a few seconds on CPU.

    Returns:
        dict with keys:
            work_dir   - pathlib.Path root of the test workspace
            csv_path   - path to the synthetic CSV
            vocab_path - path to the fitted vocab.pkl
            output_dir - path to the training output directory
    """
    work_dir = tmp_path_factory.mktemp("training_artifact_workspace")

    # Write CSV to expected relative structure (fit_tokenizer reads relative to cwd)
    data_dir = work_dir / "data" / "tabformer"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "card_transaction.v1.csv"
    csv_path.write_text(
        "\n".join(line.strip() for line in _SMOKE_CSV.splitlines())
    )

    vocab_path = data_dir / "vocab.pkl"

    # Fit tokenizer (writes vocab.pkl relative to cwd=work_dir)
    subprocess.run(
        [sys.executable, "-m", "pragma_encoder.data.fit_tokenizer"],
        cwd=str(work_dir),
        check=True,
        capture_output=True,
    )

    assert vocab_path.exists(), (
        f"fit_tokenizer did not produce {vocab_path}. "
        "Check that pragma_encoder is installed (pip install -e .)."
    )

    # Run training — one epoch, 2 customers, no max-steps cap
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run one epoch with 2 customers — no --max-steps so the epoch completes
    # and the checkpoint is written.  With limit_rows=2 + batch_size=1 this
    # takes ~2 gradient steps and finishes in a few seconds on CPU.
    result = subprocess.run(
        [
            sys.executable, "-m", "pragma_encoder.training.train",
            "--csv-path",      str(csv_path),
            "--vocab-path",    str(vocab_path),
            "--output-dir",    str(output_dir),
            "--model-variant", "pragma-s",
            "--epochs",        "1",
            "--num-workers",   "0",
            "--batch-size",    "1",
            "--limit-rows",    "2",
            "--device",        "cpu",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr (logger.info) into stdout
        text=True,
    )

    combined = result.stdout or ""
    print(combined[-2000:])  # tail for debug

    return {
        "work_dir":   work_dir,
        "csv_path":   csv_path,
        "vocab_path": vocab_path,
        "output_dir": output_dir,
        "stdout":     combined,
    }


# ===========================================================================
# 1. Artifact contract tests — checkpoint and metadata.json produced
# ===========================================================================


class TestTrainingArtifactContract:
    """Verify that training produces the required artifacts."""

    def test_checkpoint_file_produced(self, training_artifacts: dict) -> None:
        """Training must produce at least one checkpoint_epoch*.pt file.

        The checkpoint is the primary model artifact for downstream fine-tuning
        and evaluation. Its existence confirms training ran to at least 1 step.
        """
        output_dir = training_artifacts["output_dir"]
        ckpt_files = sorted(output_dir.glob("checkpoint_epoch*.pt"))
        assert len(ckpt_files) >= 1, (
            f"No checkpoint_epoch*.pt found in {output_dir}. "
            "Training must produce at least one checkpoint. "
            "Check that --checkpoint-every 1 is the default (it is)."
        )

    def test_metadata_json_produced(self, training_artifacts: dict) -> None:
        """Training must produce metadata.json in the output directory.

        metadata.json is the training artifact contract: it records the
        run configuration and the path of the final checkpoint, enabling
        reproducibility and downstream pipeline stages.
        """
        output_dir = training_artifacts["output_dir"]
        meta_path = output_dir / "metadata.json"
        assert meta_path.exists(), (
            f"metadata.json not found in {output_dir}. "
            "pragma_encoder.training.train must write metadata.json on completion."
        )

    def test_metadata_json_is_valid_json(self, training_artifacts: dict) -> None:
        """metadata.json must be valid JSON."""
        meta_path = training_artifacts["output_dir"] / "metadata.json"
        content = meta_path.read_text()
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"metadata.json is not valid JSON: {exc}\n"
                f"Content: {content[:500]}"
            )

    def test_metadata_json_has_required_top_level_keys(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json must have all required top-level keys."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        required_keys = {
            "timestamp",
            "model_variant",
            "epochs_completed",
            "global_step",
            "dataset",
            "output_dir",
            "final_checkpoint",
            "args",
        }
        missing = required_keys - set(meta.keys())
        assert not missing, (
            f"metadata.json is missing required keys: {sorted(missing)}. "
            f"Present keys: {sorted(meta.keys())}"
        )

    def test_metadata_json_records_model_variant(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json must record the correct model_variant."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        assert meta["model_variant"] == "pragma-s", (
            f"metadata.json model_variant should be 'pragma-s'. Got: {meta['model_variant']!r}"
        )

    def test_metadata_json_records_limit_rows(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json must record the limit_rows value used for the run."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        assert "dataset" in meta, "metadata.json must have a 'dataset' section."
        assert meta["dataset"].get("limit_rows") == 2, (
            f"metadata.json dataset.limit_rows should be 2 (--limit-rows 2 was passed). "
            f"Got: {meta['dataset'].get('limit_rows')!r}"
        )

    def test_metadata_json_final_checkpoint_is_valid_path(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json final_checkpoint must point to an existing file."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        final_ckpt = meta.get("final_checkpoint")
        assert final_ckpt is not None, (
            "metadata.json final_checkpoint must not be None when training produced a checkpoint."
        )
        ckpt_path = pathlib.Path(final_ckpt)
        assert ckpt_path.exists(), (
            f"metadata.json final_checkpoint={final_ckpt!r} does not exist on disk. "
            "The path must be absolute and the file must be present."
        )

    def test_metadata_json_args_section_has_limit_rows(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json args section must record limit_rows."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        args_section = meta.get("args", {})
        assert "limit_rows" in args_section, (
            "metadata.json args section must include limit_rows. "
            f"Present args keys: {sorted(args_section.keys())}"
        )


# ===========================================================================
# 2. --limit-rows argument contract
# ===========================================================================


class TestLimitRowsArg:
    """Verify the --limit-rows argument is correctly parsed."""

    def test_limit_rows_default_is_zero(self) -> None:
        """--limit-rows default must be 0 (meaning: use all rows)."""
        from pragma_encoder.training.train import parse_args  # noqa: PLC0415
        args = parse_args([])
        assert args.limit_rows == 0, (
            f"--limit-rows default should be 0 (no limit). Got: {args.limit_rows}"
        )

    def test_limit_rows_parsed_from_cli(self) -> None:
        """--limit-rows N must be parsed to args.limit_rows == N."""
        from pragma_encoder.training.train import parse_args  # noqa: PLC0415
        args = parse_args(["--limit-rows", "42"])
        assert args.limit_rows == 42, (
            f"--limit-rows 42 should give args.limit_rows == 42. Got: {args.limit_rows}"
        )

    def test_limit_rows_is_int(self) -> None:
        """--limit-rows must produce an int (not str)."""
        from pragma_encoder.training.train import parse_args  # noqa: PLC0415
        args = parse_args(["--limit-rows", "5"])
        assert isinstance(args.limit_rows, int), (
            f"args.limit_rows must be int. Got: {type(args.limit_rows)}"
        )

    def test_limit_rows_zero_means_no_cap(self) -> None:
        """--limit-rows 0 must leave args.limit_rows as 0 (no cap applied)."""
        from pragma_encoder.training.train import parse_args  # noqa: PLC0415
        args = parse_args(["--limit-rows", "0"])
        assert args.limit_rows == 0, (
            f"--limit-rows 0 should give args.limit_rows == 0. Got: {args.limit_rows}"
        )

    def test_training_stdout_confirms_limit_rows_applied(
        self, training_artifacts: dict
    ) -> None:
        """Training stdout must log that --limit-rows was applied.

        When --limit-rows N is set and N < dataset size, the training loop logs:
          '--limit-rows N: training on N customers'
        This confirms Subset was applied (not the full dataset).
        """
        stdout = training_artifacts["stdout"]
        assert "limit-rows" in stdout or "limit_rows" in stdout or "training on 2 customers" in stdout, (
            "Training stdout should log --limit-rows application. "
            f"Stdout tail (last 500 chars):\n{stdout[-500:]}"
        )


# ===========================================================================
# 3. Pipeline parameter contract
# ===========================================================================


class TestPipelineParamContract:
    """Verify pipeline functions expose the required parameters."""

    def test_pragma_pretraining_pipeline_has_max_steps(self) -> None:
        """pragma_pretraining_pipeline must have a max_steps parameter."""
        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415
        sig = inspect.signature(pragma_pretraining_pipeline)
        assert "max_steps" in sig.parameters, (
            "pragma_pretraining_pipeline must have a max_steps parameter. "
            "It is wired through to run_pretraining for smoke/tiny runs."
        )

    def test_pragma_pretraining_pipeline_max_steps_default_zero(self) -> None:
        """pragma_pretraining_pipeline max_steps default must be 0 (no early stop)."""
        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415
        sig = inspect.signature(pragma_pretraining_pipeline)
        default = sig.parameters["max_steps"].default
        assert default == 0, (
            f"pragma_pretraining_pipeline max_steps default should be 0. Got: {default!r}"
        )

    def test_pragma_pretraining_pipeline_has_limit_rows(self) -> None:
        """pragma_pretraining_pipeline must have a limit_rows parameter."""
        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415
        sig = inspect.signature(pragma_pretraining_pipeline)
        assert "limit_rows" in sig.parameters, (
            "pragma_pretraining_pipeline must have a limit_rows parameter. "
            "It is wired through to run_pretraining for smoke/tiny runs."
        )

    def test_pragma_pretraining_pipeline_limit_rows_default_zero(self) -> None:
        """pragma_pretraining_pipeline limit_rows default must be 0 (no cap)."""
        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415
        sig = inspect.signature(pragma_pretraining_pipeline)
        default = sig.parameters["limit_rows"].default
        assert default == 0, (
            f"pragma_pretraining_pipeline limit_rows default should be 0. Got: {default!r}"
        )

    def test_pragma_smoke_pipeline_has_output_dir(self) -> None:
        """pragma_smoke_training_pipeline must have an output_dir parameter."""
        from pipeline.pragma_smoke_pipeline import pragma_smoke_training_pipeline  # noqa: PLC0415
        sig = inspect.signature(pragma_smoke_training_pipeline)
        assert "output_dir" in sig.parameters, (
            "pragma_smoke_training_pipeline must have an output_dir parameter. "
            "It is passed through to pragma_smoke_training for artifact inspection."
        )

    def test_pragma_smoke_component_has_output_dir(self) -> None:
        """pragma_smoke_training component must have an output_dir parameter."""
        from pipeline.pragma_smoke_pipeline import pragma_smoke_training  # noqa: PLC0415
        sig = inspect.signature(pragma_smoke_training)
        assert "output_dir" in sig.parameters, (
            "pragma_smoke_training component must have an output_dir parameter. "
            "Training artifacts (checkpoint, metadata.json) are written to output_dir."
        )

    def test_pragma_smoke_pipeline_output_dir_default(self) -> None:
        """pragma_smoke_training_pipeline output_dir default must be a /tmp path."""
        from pipeline.pragma_smoke_pipeline import pragma_smoke_training_pipeline  # noqa: PLC0415
        sig = inspect.signature(pragma_smoke_training_pipeline)
        default = sig.parameters["output_dir"].default
        assert isinstance(default, str), (
            f"pragma_smoke_training_pipeline output_dir default must be str. Got: {type(default)}"
        )
        assert default.startswith("/tmp"), (
            f"pragma_smoke_training_pipeline output_dir default should start with /tmp "
            f"(ephemeral pod storage). Got: {default!r}"
        )


# ===========================================================================
# 4. Source contract — wheel-based entrypoint, no source-tree paths
# ===========================================================================


class TestRunPretrainingSourceContract:
    """Verify run_pretraining component uses the wheel-based training entrypoint."""

    @classmethod
    def _components_source(cls) -> str:
        import pipeline.components_pragma as mod  # noqa: PLC0415
        return pathlib.Path(inspect.getfile(mod)).read_text()

    def test_run_pretraining_uses_module_invocation(self) -> None:
        """run_pretraining must invoke training as a Python module.

        The wheel-based training image installs pragma_encoder into site-packages.
        The component must use module invocation:
            [sys.executable, '-m', 'pragma_encoder.training.train']
        Not direct file-path execution (no src/ directory in the image).
        """
        src = self._components_source()
        assert "pragma_encoder.training.train" in src, (
            "pipeline/components_pragma.py must reference pragma_encoder.training.train. "
            "run_pretraining must invoke: [sys.executable, '-m', 'pragma_encoder.training.train']"
        )

    def test_run_pretraining_has_no_src_tree_paths(self) -> None:
        """run_pretraining must not reference source-tree paths.

        The wheel image has no src/ directory. References to
        'src/pragma_encoder/...' would cause FileNotFoundError at pod startup.
        """
        src = self._components_source()
        assert "/src/pragma_encoder" not in src, (
            "pipeline/components_pragma.py must not contain '/src/pragma_encoder'. "
            "Use module invocation (python -m) — not file-path script execution."
        )

    def test_run_pretraining_uses_aws_star_env_vars(self) -> None:
        """run_pretraining must read native AWS_* env vars for S3 access.

        Native RHOAI S3 Connection schema uses AWS_* env var names.
        run_pretraining must read: AWS_S3_BUCKET, AWS_S3_ENDPOINT,
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.
        """
        src = self._components_source()
        for var in ("AWS_S3_BUCKET", "AWS_S3_ENDPOINT", "AWS_ACCESS_KEY_ID"):
            assert var in src, (
                f"pipeline/components_pragma.py must reference {var!r}. "
                "run_pretraining reads native RHOAI S3 Connection AWS_* env vars."
            )

    def test_run_pretraining_has_limit_rows_param(self) -> None:
        """run_pretraining component must have a limit_rows parameter."""
        from pipeline.components_pragma import run_pretraining  # noqa: PLC0415
        sig = inspect.signature(run_pretraining)
        assert "limit_rows" in sig.parameters, (
            "run_pretraining component must have a limit_rows parameter. "
            "It is passed as --limit-rows to pragma_encoder.training.train."
        )

    def test_run_pretraining_has_max_steps_param(self) -> None:
        """run_pretraining component must have a max_steps parameter."""
        from pipeline.components_pragma import run_pretraining  # noqa: PLC0415
        sig = inspect.signature(run_pretraining)
        assert "max_steps" in sig.parameters, (
            "run_pretraining component must have a max_steps parameter. "
            "It is passed as --max-steps to pragma_encoder.training.train."
        )

    def test_run_pretraining_passes_max_steps_to_train(self) -> None:
        """run_pretraining must wire max_steps into the training subprocess call."""
        src = self._components_source()
        assert '"--max-steps"' in src or "'--max-steps'" in src, (
            "pipeline/components_pragma.py run_pretraining must pass --max-steps to train. "
            "Check the _cmd list in run_pretraining."
        )

    def test_run_pretraining_passes_limit_rows_to_train(self) -> None:
        """run_pretraining must wire limit_rows into the training subprocess call."""
        src = self._components_source()
        assert '"--limit-rows"' in src or "'--limit-rows'" in src, (
            "pipeline/components_pragma.py run_pretraining must pass --limit-rows to train. "
            "Check the _cmd list in run_pretraining."
        )

    def test_run_pretraining_has_scratch_dir_param(self) -> None:
        """run_pretraining must have a scratch_dir parameter.

        scratch_dir makes the staging directory configurable so tests can redirect
        it to a tmpdir, and future confidential runtimes can restrict it to attested
        memory regions. It must NOT be hardcoded to /tmp/pragma-run.
        """
        from pipeline.components_pragma import run_pretraining  # noqa: PLC0415
        sig = inspect.signature(run_pretraining)
        assert "scratch_dir" in sig.parameters, (
            "run_pretraining component must have a scratch_dir parameter. "
            "The staging directory must be configurable — not hardcoded to /tmp/pragma-run."
        )

    def test_run_pretraining_scratch_dir_default_is_tmp(self) -> None:
        """run_pretraining scratch_dir default must be a /tmp path (pod-local ephemeral)."""
        from pipeline.components_pragma import run_pretraining  # noqa: PLC0415
        sig = inspect.signature(run_pretraining)
        default = sig.parameters["scratch_dir"].default
        assert isinstance(default, str), (
            f"run_pretraining scratch_dir default must be str. Got: {type(default)}"
        )
        assert default.startswith("/tmp"), (
            f"run_pretraining scratch_dir default must start with /tmp (pod-local ephemeral). "
            f"Got: {default!r}"
        )

    def test_run_pretraining_body_uses_scratch_dir_not_hardcoded(self) -> None:
        """run_pretraining body must use the scratch_dir parameter, not a hardcoded path.

        If /tmp/pragma-run is hardcoded in the function body, setting scratch_dir
        would have no effect — defeating the purpose of the parameter.
        """
        import pipeline.components_pragma as mod  # noqa: PLC0415
        src = pathlib.Path(inspect.getfile(mod)).read_text()
        # The body must NOT contain a string literal /tmp/pragma-run
        # (the default value in the signature is acceptable but the body must use the param).
        # We detect this by checking that pathlib.Path(scratch_dir) appears in the source,
        # indicating the parameter is actually used.
        assert "pathlib.Path(scratch_dir)" in src, (
            "run_pretraining body must use pathlib.Path(scratch_dir). "
            "The /tmp/pragma-run path must come from the parameter, not be hardcoded."
        )

    def test_run_pretraining_passes_dataset_name_to_train(self) -> None:
        """run_pretraining must wire --dataset-name into the training subprocess call.

        --dataset-name passes the original S3 URI or dataset label to the wheel
        so it is recorded in metadata.json, not the local staging path.
        """
        src = self._components_source()
        assert '"--dataset-name"' in src or "'--dataset-name'" in src, (
            "pipeline/components_pragma.py run_pretraining must pass --dataset-name to train. "
            "Check the _cmd list in run_pretraining."
        )

    def test_run_pretraining_uploads_metadata(self) -> None:
        """run_pretraining must upload metadata.json to S3 alongside the checkpoint.

        metadata.json must be durable — not ephemeral pod-local storage only.
        It is uploaded to S3 at the same prefix as the checkpoint.
        """
        src = self._components_source()
        assert "metadata.json" in src, (
            "pipeline/components_pragma.py run_pretraining must reference metadata.json. "
            "metadata.json must be uploaded to S3 alongside the checkpoint."
        )


# ===========================================================================
# 5. Metadata staging path and dataset reference contract
# ===========================================================================


class TestMetadataStagingPath:
    """Verify that metadata.json records the staging path and dataset reference correctly."""

    def test_metadata_dataset_has_csv_staging_path(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json dataset section must have csv_staging_path (not csv_path).

        csv_staging_path makes explicit that this is the job-local staging path,
        not the original dataset source (which is recorded in dataset_name).
        """
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        assert "csv_staging_path" in meta["dataset"], (
            "metadata.json dataset section must have 'csv_staging_path' key. "
            f"Present dataset keys: {sorted(meta['dataset'].keys())}"
        )

    def test_metadata_dataset_no_raw_csv_path_key(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json dataset section must NOT have the old 'csv_path' key.

        The field was renamed to 'csv_staging_path' to make its meaning unambiguous.
        If 'csv_path' is still present, the rename did not take effect.
        """
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        assert "csv_path" not in meta["dataset"], (
            "metadata.json dataset section must not have 'csv_path'. "
            "The field was renamed to 'csv_staging_path'. "
            f"Present dataset keys: {sorted(meta['dataset'].keys())}"
        )

    def test_metadata_dataset_has_dataset_name(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json dataset section must have dataset_name.

        dataset_name records the original dataset reference (S3 URI or label).
        For local runs without --dataset-name, it falls back to csv_staging_path value.
        """
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        assert "dataset_name" in meta["dataset"], (
            "metadata.json dataset section must have 'dataset_name' key. "
            f"Present dataset keys: {sorted(meta['dataset'].keys())}"
        )

    def test_metadata_dataset_name_is_string(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json dataset_name must be a non-empty string."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        name = meta["dataset"].get("dataset_name")
        assert isinstance(name, str) and name, (
            f"metadata.json dataset.dataset_name must be a non-empty string. Got: {name!r}"
        )

    def test_metadata_csv_staging_path_is_absolute(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json csv_staging_path must be an absolute path."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        staging_path = meta["dataset"]["csv_staging_path"]
        assert pathlib.Path(staging_path).is_absolute(), (
            f"metadata.json csv_staging_path must be absolute. Got: {staging_path!r}"
        )

    def test_dataset_name_arg_default_is_empty_string(self) -> None:
        """--dataset-name default must be empty string (set by pipeline, not user)."""
        from pragma_encoder.training.train import parse_args  # noqa: PLC0415
        args = parse_args([])
        assert args.dataset_name == "", (
            f"--dataset-name default must be '' (empty). Got: {args.dataset_name!r}"
        )

    def test_dataset_name_arg_parsed_from_cli(self) -> None:
        """--dataset-name must be passed through to args.dataset_name."""
        from pragma_encoder.training.train import parse_args  # noqa: PLC0415
        args = parse_args(["--dataset-name", "s3://mybucket/pragma-encoder/data/tabformer/"])
        assert args.dataset_name == "s3://mybucket/pragma-encoder/data/tabformer/", (
            f"--dataset-name not correctly parsed. Got: {args.dataset_name!r}"
        )


# ===========================================================================
# 6. Metadata secrets / credential redaction contract
# ===========================================================================


class TestMetadataSecrets:
    """Verify that metadata.json never contains credentials or sensitive values.

    AWS_* credentials are env vars, not CLI args — they must never appear in
    metadata.json. The _METADATA_ALLOWED_ARG_KEYS allowlist enforces this
    at the source level; these tests enforce it at the artifact level.
    """

    def test_metadata_args_no_access_key_id(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json args section must not contain AWS_ACCESS_KEY_ID key."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        args_section = meta.get("args", {})
        assert "aws_access_key_id" not in args_section, (
            "metadata.json args must not contain 'aws_access_key_id'. "
            "AWS credentials are env vars, not CLI args."
        )
        assert "AWS_ACCESS_KEY_ID" not in args_section, (
            "metadata.json args must not contain 'AWS_ACCESS_KEY_ID'. "
            "AWS credentials are env vars, not CLI args."
        )

    def test_metadata_args_no_secret_access_key(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json args section must not contain AWS_SECRET_ACCESS_KEY key."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        args_section = meta.get("args", {})
        assert "aws_secret_access_key" not in args_section, (
            "metadata.json args must not contain 'aws_secret_access_key'. "
        )
        assert "AWS_SECRET_ACCESS_KEY" not in args_section, (
            "metadata.json args must not contain 'AWS_SECRET_ACCESS_KEY'. "
        )

    def test_metadata_no_top_level_credentials(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json top-level keys must not include credential-like names."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        credential_patterns = {
            "access_key", "secret_key", "password", "token",
            "aws_access_key_id", "aws_secret_access_key",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
        }
        found = credential_patterns & set(meta.keys())
        assert not found, (
            f"metadata.json top-level keys contain credential-like names: {found}. "
            "Credentials must never be recorded in metadata.json."
        )

    def test_metadata_args_only_allowed_keys(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json args section must only contain keys from the allowed list.

        The _METADATA_ALLOWED_ARG_KEYS allowlist in train.py defines exactly
        which CLI args are safe to record. This test confirms the allowlist is
        applied and no unexpected keys slip through.
        """
        from pragma_encoder.training.train import _METADATA_ALLOWED_ARG_KEYS  # noqa: PLC0415
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        args_section = meta.get("args", {})
        unexpected = set(args_section.keys()) - _METADATA_ALLOWED_ARG_KEYS
        assert not unexpected, (
            f"metadata.json args section has unexpected keys: {sorted(unexpected)}. "
            f"Allowed keys: {sorted(_METADATA_ALLOWED_ARG_KEYS)}. "
            "Add to _METADATA_ALLOWED_ARG_KEYS only if the value cannot be a credential."
        )

    def test_metadata_args_excludes_csv_path(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json args must not include csv_path (staging path, not dataset ref).

        csv_path is deliberately excluded from _METADATA_ALLOWED_ARG_KEYS — it is
        a staging/local path that is already captured in dataset.csv_staging_path.
        Including it in args would duplicate (and potentially confuse) the record.
        """
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        args_section = meta.get("args", {})
        assert "csv_path" not in args_section, (
            "metadata.json args must not include 'csv_path'. "
            "It is excluded from _METADATA_ALLOWED_ARG_KEYS. "
            "The staging path is recorded in dataset.csv_staging_path."
        )

    def test_metadata_args_excludes_vocab_path(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json args must not include vocab_path (staging path)."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        args_section = meta.get("args", {})
        assert "vocab_path" not in args_section, (
            "metadata.json args must not include 'vocab_path'. "
            "It is excluded from _METADATA_ALLOWED_ARG_KEYS."
        )

    def test_metadata_args_excludes_device(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json args must not include device (not reproducibility-relevant)."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        args_section = meta.get("args", {})
        assert "device" not in args_section, (
            "metadata.json args must not include 'device'. "
            "It is excluded from _METADATA_ALLOWED_ARG_KEYS."
        )

    def test_safe_args_for_metadata_function_exists(self) -> None:
        """_safe_args_for_metadata function must be importable from train module."""
        from pragma_encoder.training.train import _safe_args_for_metadata  # noqa: PLC0415
        assert callable(_safe_args_for_metadata), (
            "_safe_args_for_metadata must be a callable function in pragma_encoder.training.train."
        )

    def test_safe_args_for_metadata_excludes_credential_keys(self) -> None:
        """_safe_args_for_metadata must exclude credential-adjacent keys even if present in args."""
        import argparse  # noqa: PLC0415

        from pragma_encoder.training import train as _train_mod  # noqa: PLC0415
        # Build a namespace with a fake credential key
        fake_args = argparse.Namespace(
            model_variant="pragma-s",
            epochs=1,
            batch_size=32,
            lr=1e-4,
            max_steps=0,
            limit_rows=0,
            seed=42,
            num_workers=4,
            checkpoint_every=1,
            s3_checkpoint_prefix="",
            dataset_name="test-dataset",
            # These should NOT appear in output:
            csv_path="/tmp/data.csv",
            vocab_path="/tmp/vocab.pkl",
            output_dir="/tmp/output",
            device="cpu",
            resume=False,
        )
        result = _train_mod._safe_args_for_metadata(fake_args)
        _keys = sorted(result.keys())
        assert "csv_path" not in result, (
            f"_safe_args_for_metadata must exclude 'csv_path'. Got keys: {_keys}"
        )
        assert "vocab_path" not in result, (
            f"_safe_args_for_metadata must exclude 'vocab_path'. Got keys: {_keys}"
        )
        assert "device" not in result, (
            f"_safe_args_for_metadata must exclude 'device'. Got keys: {_keys}"
        )
        assert "model_variant" in result, (
            f"_safe_args_for_metadata must include 'model_variant'. Got keys: {_keys}"
        )
        _got = result.get("dataset_name")
        assert _got == "test-dataset", (
            f"_safe_args_for_metadata should include dataset_name. Got: {_got!r}"
        )


# ===========================================================================
# 7. metrics.jsonl contract
# ===========================================================================


class TestMetricsJsonlContract:
    """Verify that training produces a valid metrics.jsonl file.

    metrics.jsonl is the structured training metrics artifact that enables:
      - Loss curve plotting (pragma_encoder.evaluation.plot_loss)
      - Pipeline artifact publication to object storage
      - Downstream run comparison and reproducibility

    Every step-level record must have: step, epoch, train_loss, learning_rate,
    timestamp, checkpoint_path.  Epoch-end and checkpoint event records have
    an 'event' key.  All records must be valid JSON objects.
    """

    def test_metrics_jsonl_produced(self, training_artifacts: dict) -> None:
        """Training must produce metrics.jsonl in the output directory."""
        output_dir = training_artifacts["output_dir"]
        metrics_path = output_dir / "metrics.jsonl"
        assert metrics_path.exists(), (
            f"metrics.jsonl not found in {output_dir}. "
            "pragma_encoder.training.train must write metrics.jsonl at each training step."
        )

    def test_metrics_jsonl_is_non_empty(self, training_artifacts: dict) -> None:
        """metrics.jsonl must contain at least one record."""
        metrics_path = training_artifacts["output_dir"] / "metrics.jsonl"
        lines = [ln.strip() for ln in metrics_path.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 1, (
            "metrics.jsonl must contain at least one record. "
            "Training ran with limit_rows=2 — at least one gradient step must occur."
        )

    def test_metrics_jsonl_each_line_is_valid_json(
        self, training_artifacts: dict
    ) -> None:
        """Every line in metrics.jsonl must be valid JSON."""
        metrics_path = training_artifacts["output_dir"] / "metrics.jsonl"
        for i, line in enumerate(metrics_path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(
                    f"metrics.jsonl line {i} is not valid JSON: {exc}\n"
                    f"Content: {line[:200]}"
                )

    def test_metrics_jsonl_step_records_have_required_keys(
        self, training_artifacts: dict
    ) -> None:
        """Step-level records must have step, epoch, train_loss, learning_rate, timestamp."""
        metrics_path = training_artifacts["output_dir"] / "metrics.jsonl"
        required = {"step", "epoch", "train_loss", "learning_rate", "timestamp"}
        step_records = [
            json.loads(ln)
            for ln in metrics_path.read_text().splitlines()
            if ln.strip() and "event" not in json.loads(ln)
            and "train_loss" in json.loads(ln)
        ]
        assert step_records, (
            "metrics.jsonl must contain at least one step-level record "
            "(a record with 'train_loss' and without 'event')."
        )
        for record in step_records:
            missing = required - set(record.keys())
            assert not missing, (
                f"Step-level record missing keys: {sorted(missing)}. "
                f"Record: {record}"
            )

    def test_metrics_jsonl_train_loss_is_positive_float(
        self, training_artifacts: dict
    ) -> None:
        """Step-level train_loss values must be positive floats."""
        metrics_path = training_artifacts["output_dir"] / "metrics.jsonl"
        for line in metrics_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "train_loss" in record and "event" not in record:
                loss = record["train_loss"]
                assert isinstance(loss, (int, float)), (
                    f"train_loss must be a number. Got: {type(loss)} ({loss!r})"
                )
                assert loss > 0, (
                    f"train_loss must be positive (meaningful loss). Got: {loss}"
                )

    def test_metrics_jsonl_steps_are_monotonically_increasing(
        self, training_artifacts: dict
    ) -> None:
        """Step values in step-level records must be strictly increasing."""
        metrics_path = training_artifacts["output_dir"] / "metrics.jsonl"
        steps = [
            json.loads(ln)["step"]
            for ln in metrics_path.read_text().splitlines()
            if ln.strip() and "train_loss" in json.loads(ln) and "event" not in json.loads(ln)
        ]
        assert steps == sorted(steps), (
            f"Step values in metrics.jsonl must be non-decreasing. Got: {steps}"
        )

    def test_metadata_json_records_metrics_jsonl_path(
        self, training_artifacts: dict
    ) -> None:
        """metadata.json must record the path to metrics.jsonl."""
        meta = json.loads((training_artifacts["output_dir"] / "metadata.json").read_text())
        assert "metrics_jsonl" in meta, (
            "metadata.json must have a 'metrics_jsonl' key pointing to the metrics file. "
            f"Present keys: {sorted(meta.keys())}"
        )
        assert meta["metrics_jsonl"], (
            "metadata.json 'metrics_jsonl' must be a non-empty string path."
        )
