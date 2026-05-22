"""Level 2b — OpenShift AI runtime image contract probe.

Purpose:
  Prove that the two-image contract holds at runtime in the cluster:
    - Workbench image: kfp + kfp-kubernetes importable
    - Training image: pragma_encoder.model, pragma_encoder.data importable
    - Training image: kfp-kubernetes is present (inherited from workbench base)
      but src/pragma_encoder/training/checkpoints.py does NOT require it at import time
    - Training image: no runtime git clone required

These tests are read-only from the cluster perspective (namespace, RBAC,
DSPA are never modified). Ephemeral Job probes are created with test labels
and cleaned up by the cleanup_labelled_resources fixture.

Guard variables:
  RUN_OPENSHIFT_TESTS=1           — suite-wide (conftest)
  RUN_OPENSHIFT_IMAGE_CONTRACT=1  — enables all Level 2b tests
  PRAGMA_TEST_NAMESPACE=<ns>      — required (conftest)
  PRAGMA_TRAINING_IMAGE=<img>     — required for runtime probe tests
  PRAGMA_WORKBENCH_IMAGE=<img>    — optional; if unset, workbench probe is skipped

Architecture boundary:
  Level 2b proves image build-time and runtime contracts.
  Level 2b does NOT prove KFP pipeline execution (Level 3).
  Level 2b does NOT prove PyTorchJob DDP (Level 4).

Safety:
  - All created resources carry both test labels (test-run + test-id)
  - cleanup_labelled_resources fixture deletes only label-scoped resources
  - No secrets are printed or asserted on
  - No namespace mutations
"""

from __future__ import annotations

import os
import pathlib
import time

import pytest

from tests.openshift.oc import oc, oc_json, redact

# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

_IMAGE_CONTRACT_ENABLED = os.environ.get("RUN_OPENSHIFT_IMAGE_CONTRACT") == "1"

_require_image_contract = pytest.mark.skipif(
    not _IMAGE_CONTRACT_ENABLED,
    reason=(
        "Image contract runtime probe is opt-in. "
        "Set RUN_OPENSHIFT_IMAGE_CONTRACT=1 to enable. "
        "Also requires RUN_OPENSHIFT_TESTS=1 and PRAGMA_TEST_NAMESPACE=<namespace>."
    ),
)

# ---------------------------------------------------------------------------
# Local (no cluster) prerequisite checks
# ---------------------------------------------------------------------------


class TestImageContractLocalPrereqs:
    """Local checks that do not require cluster access.

    Run without any opt-in env var.
    These catch regressions in the image build files before any cluster Job is created.
    """

    _REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
    _DOCKERFILE_TRAINING = _REPO_ROOT / "openshift" / "training" / "Dockerfile.training"
    _NOTEBOOK_REQUIREMENTS = _REPO_ROOT / "openshift" / "notebook-image" / "requirements.txt"
    _PYPROJECT = _REPO_ROOT / "pyproject.toml"
    _CHECKPOINTS_PY = _REPO_ROOT / "src" / "pragma_encoder" / "training" / "checkpoints.py"

    def test_dockerfile_training_exists(self) -> None:
        """openshift/training/Dockerfile.training must exist."""
        assert self._DOCKERFILE_TRAINING.exists(), (
            f"Dockerfile.training not found at {self._DOCKERFILE_TRAINING}. "
            "This file is required to build the training image."
        )

    def test_dockerfile_training_copies_scripts(self) -> None:
        """Dockerfile.training must COPY scripts/ into the image."""
        text = self._DOCKERFILE_TRAINING.read_text()
        copy_lines = [ln for ln in text.splitlines() if "COPY" in ln and "scripts/" in ln]
        assert copy_lines, (
            "Dockerfile.training must COPY scripts/ into the image. "
            "train_pragma.py is invoked from scripts/. "
            "See docs/openshift-image-contract.md."
        )

    def test_dockerfile_training_validates_imports_at_build(self) -> None:
        """Dockerfile.training must include a RUN python -c 'import pragma_encoder.*' validation step."""
        text = self._DOCKERFILE_TRAINING.read_text()
        has_import_check = "python" in text and "import pragma_encoder" in text
        assert has_import_check, (
            "Dockerfile.training must run 'python -c \"import pragma_encoder.*\"' at build time. "
            "This catches missing __init__.py or broken imports at image build, "
            "not at job runtime. See docs/openshift-image-contract.md."
        )

    def test_checkpoints_module_exists(self) -> None:
        """src/training/checkpoints.py must exist for the S3 resume contract."""
        assert self._CHECKPOINTS_PY.exists(), (
            f"src/training/checkpoints.py not found at {self._CHECKPOINTS_PY}. "
            "This module implements the all-rank S3 download pattern (TD-006 fix). "
            "Create it before this test can pass."
        )

    def test_checkpoints_module_no_kfp_import(self) -> None:
        """src/training/checkpoints.py must not import kfp or kfp_kubernetes."""
        if not self._CHECKPOINTS_PY.exists():
            pytest.skip("src/training/checkpoints.py does not exist yet.")
        text = self._CHECKPOINTS_PY.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if stripped.startswith("import kfp") or stripped.startswith("from kfp"):
                pytest.fail(
                    f"src/training/checkpoints.py must not import kfp. "
                    f"Found: {line!r}. "
                    "S3 checkpoint/resume is a training-image concern — "
                    "kfp belongs in the workbench compile environment only."
                )
            if "kfp_kubernetes" in stripped and (
                stripped.startswith("import") or stripped.startswith("from")
            ):
                pytest.fail(
                    f"src/training/checkpoints.py must not import kfp_kubernetes. "
                    f"Found: {line!r}. "
                    "kfp-kubernetes belongs in the workbench compile environment only."
                )

    def test_notebook_requirements_has_both_kfp_packages(self) -> None:
        """openshift/notebook-image/requirements.txt must have kfp and kfp-kubernetes."""
        text = self._NOTEBOOK_REQUIREMENTS.read_text()
        lines = [ln.strip() for ln in text.splitlines()]
        has_kfp = any(ln.startswith("kfp") and not ln.startswith("kfp-kubernetes") for ln in lines)
        has_kfp_k8s = any(ln.startswith("kfp-kubernetes") for ln in lines)
        assert has_kfp, "notebook requirements.txt must include kfp."
        assert has_kfp_k8s, "notebook requirements.txt must include kfp-kubernetes."


# ---------------------------------------------------------------------------
# Runtime probe — training image
# ---------------------------------------------------------------------------


class TestTrainingImageRuntimeContract:
    """Opt-in cluster probe: verify training image imports at runtime.

    Creates a labelled batch/v1 Job that runs a python import check in the
    training image. The Job exits 0 if all required imports succeed.

    Requires:
      RUN_OPENSHIFT_IMAGE_CONTRACT=1
      RUN_OPENSHIFT_TESTS=1
      PRAGMA_TRAINING_IMAGE=<image>
      PRAGMA_TEST_NAMESPACE=<namespace>
    """

    _PROBE_TIMEOUT = 120  # seconds

    @_require_image_contract
    def test_training_image_core_imports_succeed(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        cleanup_labelled_resources: None,
    ) -> None:
        """Training image must import pragma_encoder.model, pragma_encoder.data at runtime.

        Creates a one-off batch/v1 Job that runs:
          python -c "import pragma_encoder.model; import pragma_encoder.data; print('OK')"

        The job must exit 0.
        Note: pragma_encoder.workbench is NOT checked — it lives in
        tools/workbench/ and is not part of the installed wheel.
        """
        image = os.environ.get("PRAGMA_TRAINING_IMAGE", "").strip()
        if not image:
            pytest.skip(
                "PRAGMA_TRAINING_IMAGE is not set. "
                "Set it to the training image URI to run this test."
            )

        job_name = f"pragma-img-core-{test_id}"
        _run_import_probe(
            job_name=job_name,
            namespace=runtime_namespace,
            image=image,
            test_id=test_id,
            import_check=(
                "import pragma_encoder.model; "
                "import pragma_encoder.data; "
                "import pragma_encoder.training.checkpoints; "
                "print('PRAGMA core imports OK')"
            ),
            expected_marker="PRAGMA core imports OK",
            timeout=self._PROBE_TIMEOUT,
        )

    @_require_image_contract
    def test_training_image_checkpoints_module_importable_without_kfp(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        cleanup_labelled_resources: None,
    ) -> None:
        """Training image: src.training.checkpoints must import without kfp installed.

        This verifies the KFP isolation boundary at runtime:
          - src.training.checkpoints is importable
          - It does NOT import kfp or kfp_kubernetes at module load time

        The probe temporarily hides kfp_kubernetes by moving it in sys.modules,
        then imports src.training.checkpoints to confirm no ImportError.
        """
        image = os.environ.get("PRAGMA_TRAINING_IMAGE", "").strip()
        if not image:
            pytest.skip("PRAGMA_TRAINING_IMAGE is not set.")

        job_name = f"pragma-img-ckpt-{test_id}"
        _run_import_probe(
            job_name=job_name,
            namespace=runtime_namespace,
            image=image,
            test_id=test_id,
            import_check=(
                "import sys; "
                "sys.modules['kfp_kubernetes'] = None; "  # Simulate absence
                "import pragma_encoder.training.checkpoints; "
                "print('checkpoints importable without kfp_kubernetes OK')"
            ),
            expected_marker="checkpoints importable without kfp_kubernetes OK",
            timeout=self._PROBE_TIMEOUT,
        )


# ---------------------------------------------------------------------------
# Helper: submit a probe Job and assert log marker
# ---------------------------------------------------------------------------


def _run_import_probe(
    job_name: str,
    namespace: str,
    image: str,
    test_id: str,
    import_check: str,
    expected_marker: str,
    timeout: int,
) -> None:
    """Submit a one-off batch/v1 Job running a python import check.

    Args:
        job_name:        Kubernetes Job name.
        namespace:       Namespace to create the Job in.
        image:           Container image URI.
        test_id:         Test ID for label scoping.
        import_check:    Python one-liner to run in -c.
        expected_marker: String that must appear in pod logs.
        timeout:         Seconds to wait for Job completion.
    """
    import json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {
                "pragma.redhat.com/test-run": "true",
                "pragma.redhat.com/test-id": test_id,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {
                    "labels": {
                        "pragma.redhat.com/test-run": "true",
                        "pragma.redhat.com/test-id": test_id,
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "probe",
                            "image": image,
                            "imagePullPolicy": "Always",
                            "command": ["python", "-c", import_check],
                            "resources": {
                                "requests": {"cpu": "200m", "memory": "512Mi"},
                                "limits": {"cpu": "1", "memory": "1Gi"},
                            },
                        }
                    ],
                },
            },
        },
    }

    import tempfile  # noqa: F811,PLC0415
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(manifest, f)
        manifest_path = f.name

    oc(["apply", "-f", manifest_path], namespace=namespace)
    print(f"\n[Level 2b] Job submitted: {job_name}")

    # Wait for Job completion
    deadline = time.time() + timeout
    completed = False
    while time.time() < deadline:
        try:
            job_json = oc_json(["get", "job", job_name], namespace=namespace, timeout=15)
            status = job_json.get("status", {})
            if status.get("succeeded", 0) >= 1:
                completed = True
                print(f"[Level 2b] Job {job_name}: Succeeded")
                break
            if status.get("failed", 0) >= 1:
                # Collect logs before failing
                logs = oc(
                    ["logs", f"job/{job_name}", "--tail", "50"],
                    namespace=namespace,
                    check=False,
                    timeout=30,
                )
                pytest.fail(
                    f"Job {job_name!r} failed. "
                    f"Logs (redacted):\n{redact(logs.stdout)}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[Level 2b] status poll error (retrying): {redact(str(exc))}")
        time.sleep(5)

    if not completed:
        pytest.fail(
            f"Job {job_name!r} did not complete within {timeout}s. "
            f"Check: oc describe job {job_name} -n {namespace}"
        )

    # Collect and check logs
    log_result = oc(
        ["logs", f"job/{job_name}", "--tail", "100"],
        namespace=namespace,
        check=False,
        timeout=30,
    )
    logs = redact(log_result.stdout)
    assert expected_marker in logs, (
        f"Expected marker {expected_marker!r} not found in pod logs. "
        f"Logs:\n{logs}"
    )
    print(f"[Level 2b] Marker confirmed: {expected_marker!r}")
