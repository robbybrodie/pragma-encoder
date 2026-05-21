"""Local tests for the Level 5 S3 checkpoint/resume manifest renderer and prereqs.

These tests run by default (no RUN_OPENSHIFT_TESTS or cluster needed):

  pytest tests/test_s3_manifest_render.py

They were extracted from tests/openshift/test_05_s3_checkpoint_resume.py so
that the YAML-render and local-prereq checks are always run in CI without
requiring the OpenShift opt-in flag.

The real cluster runtime smoke (TestS3CheckpointResumeSmoke) remains in
tests/openshift/test_05_s3_checkpoint_resume.py under the OpenShift gate.
To run the full two-run S3 smoke on a live cluster:

  RUN_OPENSHIFT_TESTS=1 \\
  RUN_OPENSHIFT_S3_RESUME_SMOKE=1 \\
  PRAGMA_TEST_NAMESPACE=<namespace> \\
  PRAGMA_TRAINING_IMAGE=<image> \\
  pytest tests/openshift/test_05_s3_checkpoint_resume.py
"""

from __future__ import annotations

import pathlib
import re

# Import the manifest renderer from the Level 5 cluster test file.
# This import is safe at collection time — it only executes stdlib code.
from tests.openshift.test_05_s3_checkpoint_resume import _render_s3_resume_manifest

# ---------------------------------------------------------------------------
# Mirror the constants from test_05 so local tests are self-contained.
# These must match the values in test_05_s3_checkpoint_resume.py.
# ---------------------------------------------------------------------------

_S3_RESUME_JOB_PREFIX = "pragma-s3"   # job name prefix (must match test_05)
_DNS_LABEL_LIMIT = 63                  # RFC 1035 §2.3.4 — max DNS label length


# ===========================================================================
# 1. Local prerequisite checks (no cluster, no S3, no oc)
# ===========================================================================


class TestS3ResumeLocalPrereqs:
    """Local checks that verify the Level 5 implementation is present.

    Run by default (no opt-in flag required). These tests confirm that:
      - The job-name prefix is safe for RFC 1035 DNS label limits.
      - src/pragma_encoder/training/checkpoints.py exists (TD-006 fix module).
      - scripts/train_pragma.py calls resolve_resume_checkpoint (integration).

    No cluster, no S3 credentials, no oc binary required.
    """

    def test_s3_resume_job_prefix_safe(self) -> None:
        """_S3_RESUME_JOB_PREFIX must leave room for test_id + KFTO master suffix.

        Level 5 job names are constructed as:
            <prefix>-r1-<test_id>-master-0   (rank 0 pod)

        The full name must stay ≤ 63 chars (RFC 1035 §2.3.4).
        A typical test_id is 35 chars (pragma-it-YYYYMMDD-HHMMSS-xxxxxxxx).
        KFTO appends '-master-0' (9 chars) to the job name.
        """
        typical_test_id_len = 35  # pragma-it-YYYYMMDD-HHMMSS-xxxxxxxx
        kfto_master_suffix_len = len("-master-0")
        max_prefix = _DNS_LABEL_LIMIT - typical_test_id_len - 1 - kfto_master_suffix_len
        assert len(_S3_RESUME_JOB_PREFIX) <= max_prefix, (
            f"_S3_RESUME_JOB_PREFIX {_S3_RESUME_JOB_PREFIX!r} is "
            f"{len(_S3_RESUME_JOB_PREFIX)} chars. "
            f"Max safe prefix: {max_prefix} chars."
        )

    def test_checkpoints_module_exists(self) -> None:
        """src/pragma_encoder/training/checkpoints.py must exist.

        This module implements the all-rank S3 download pattern that fixes
        TD-006 (multi-node checkpoint resume with per-pod emptyDir).
        """
        checkpoints_path = (
            pathlib.Path(__file__).parent.parent
            / "src" / "pragma_encoder" / "training" / "checkpoints.py"
        )
        assert checkpoints_path.exists(), (
            "src/pragma_encoder/training/checkpoints.py not found. "
            "The TD-006 fix requires this module. "
            "Implement it before the Level 5 runtime smoke can pass."
        )

    def test_train_pragma_has_resolve_resume_checkpoint(self) -> None:
        """scripts/train_pragma.py must call resolve_resume_checkpoint.

        This confirms TD-006 is integrated into the training script:
        all ranks download the checkpoint from S3, not just rank 0.
        """
        train_script = (
            pathlib.Path(__file__).parent.parent / "scripts" / "train_pragma.py"
        )
        text = train_script.read_text()
        assert "resolve_resume_checkpoint" in text, (
            "scripts/train_pragma.py must call resolve_resume_checkpoint() "
            "from src/pragma_encoder/training/checkpoints.py. "
            "The TD-006 fix requires all ranks to download the checkpoint independently."
        )


# ===========================================================================
# 2. Manifest-render tests (no cluster, no oc, just yaml.safe_load)
# ===========================================================================


class TestS3ResumeManifestRender:
    """Local manifest-render tests — no cluster access required.

    Calls _render_s3_resume_manifest() and parses the result with
    yaml.safe_load to verify structural correctness without a cluster.

    Catches regressions like the YAML indentation bug that previously caused
    oc apply to fail when --resume was embedded in a block scalar.

    8 tests — always run (no guard variable needed).
    """

    # Common parameters shared by all render calls.
    _DEFAULTS = dict(
        job_name="pragma-s3-render-test",
        namespace="test-ns",
        image="image-registry.example.com/test/pragma-training:latest",
        test_id="pragma-it-20260521-120000-deadbeef",
        nnodes=2,
        s3_prefix="pragma-encoder/test-checkpoints/pragma-it-20260521",
        max_steps=5,
    )

    def test_manifest_is_valid_yaml(self) -> None:
        """_render_s3_resume_manifest must produce parseable YAML."""
        import yaml  # noqa: PLC0415

        manifest = _render_s3_resume_manifest(**self._DEFAULTS, resume=False)
        parsed = yaml.safe_load(manifest)
        assert parsed is not None, (
            "_render_s3_resume_manifest must produce non-empty YAML. "
            f"Got: {manifest[:200]!r}"
        )

    def test_manifest_kind_is_pytorchjob(self) -> None:
        """Rendered manifest kind must be PyTorchJob."""
        import yaml  # noqa: PLC0415

        parsed = yaml.safe_load(_render_s3_resume_manifest(**self._DEFAULTS, resume=False))
        assert parsed["kind"] == "PyTorchJob", (
            f"Expected kind=PyTorchJob, got: {parsed.get('kind')!r}"
        )

    def test_manifest_api_version_is_kubeflow_v1(self) -> None:
        """Rendered manifest apiVersion must be kubeflow.org/v1."""
        import yaml  # noqa: PLC0415

        parsed = yaml.safe_load(_render_s3_resume_manifest(**self._DEFAULTS, resume=False))
        assert parsed["apiVersion"] == "kubeflow.org/v1", (
            f"Expected apiVersion=kubeflow.org/v1, got: {parsed.get('apiVersion')!r}"
        )

    def test_manifest_has_master_and_worker(self) -> None:
        """Rendered manifest must define both Master and Worker replica specs."""
        import yaml  # noqa: PLC0415

        parsed = yaml.safe_load(_render_s3_resume_manifest(**self._DEFAULTS, resume=False))
        specs = parsed["spec"]["pytorchReplicaSpecs"]
        assert "Master" in specs, (
            "Rendered manifest must define Master replica spec. "
            f"Found specs: {list(specs.keys())}"
        )
        assert "Worker" in specs, (
            "Rendered manifest must define Worker replica spec. "
            f"Found specs: {list(specs.keys())}"
        )

    def test_worker_replicas_equals_nnodes_minus_one(self) -> None:
        """Worker replicas must equal nnodes - 1 for correct N-node topology."""
        import yaml  # noqa: PLC0415

        for nnodes in (2, 3, 4):
            params = {**self._DEFAULTS, "nnodes": nnodes}
            parsed = yaml.safe_load(_render_s3_resume_manifest(**params, resume=False))
            worker_replicas = parsed["spec"]["pytorchReplicaSpecs"]["Worker"]["replicas"]
            assert worker_replicas == nnodes - 1, (
                f"nnodes={nnodes}: Worker replicas must be nnodes-1={nnodes - 1}. "
                f"Got: {worker_replicas}"
            )

    def test_manifest_contains_s3_prefix(self) -> None:
        """Rendered manifest must include the --s3-checkpoint-prefix argument."""
        s3_prefix = "pragma-encoder/test-checkpoints/pragma-it-20260521"
        params = {**self._DEFAULTS, "s3_prefix": s3_prefix}
        manifest = _render_s3_resume_manifest(**params, resume=False)
        assert f"--s3-checkpoint-prefix {s3_prefix}" in manifest, (
            f"Manifest must contain '--s3-checkpoint-prefix {s3_prefix}'. "
            "The train_pragma.py S3 upload/download depends on this argument."
        )

    def test_resume_manifest_contains_resume_flag(self) -> None:
        """Resume manifest must include --resume on the same command line."""
        manifest = _render_s3_resume_manifest(**self._DEFAULTS, resume=True)
        assert "--resume" in manifest, (
            "resume=True manifest must contain '--resume'. "
            "Without it, train_pragma.py will not attempt to load a checkpoint."
        )

    def test_non_resume_manifest_lacks_resume_flag(self) -> None:
        """Non-resume manifest must not contain --resume.

        Prevents false-positive resume attempts on first-run training jobs
        where no checkpoint exists yet. Uses word-boundary regex to avoid
        matching '--s3-checkpoint-prefix' (which contains 'resume' in its name).
        """
        manifest = _render_s3_resume_manifest(**self._DEFAULTS, resume=False)
        resume_as_flag = re.search(r"(?<!\w)--resume(?!\w)", manifest)
        assert resume_as_flag is None, (
            "resume=False manifest must not contain '--resume'. "
            f"Found match at position {resume_as_flag.start() if resume_as_flag else 'N/A'}."
        )
