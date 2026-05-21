"""PRAGMA workbench DSPA/KFP v2 submit path.

Implements the compile → upload → run creation plumbing for submitting
a compiled PRAGMA pipeline YAML to the OpenShift AI Data Science Pipelines
(DSPA) / KFP v2 API from inside the PRAGMA workbench pod.

This module does NOT import kfp at module level. kfp is an optional
dependency, imported lazily inside make_kfp_client(). This allows the
module to load in any environment without requiring kfp to be installed.

Public API:
    get_dspa_endpoint(...)      — resolve KFP API endpoint URL
    get_service_account_token() — read SA token from pod mount; never printed
    DSPAConfig                  — lightweight endpoint + auth config value object
    make_kfp_client(...)        — construct kfp.Client (lazy kfp import)
    upload_pipeline(...)        — upload compiled YAML to DSPA
    submit_pipeline_run(...)    — create a KFP pipeline run on DSPA

Endpoint resolution order (get_dspa_endpoint):
    1. PRAGMA_DSPA_ENDPOINT env var (verbatim URL — any value accepted)
    2. namespace kwarg passed directly to get_dspa_endpoint()
    3. PRAGMA_DSPA_NAMESPACE env var
    4. PRAGMA_TEST_NAMESPACE env var
    5. SA namespace file (/var/run/secrets/kubernetes.io/serviceaccount/namespace)
    6. RuntimeError — no namespace source is available

Authentication model:
    Port 8888 is the direct KFP API server, not behind an OAuth proxy.
    The SA token is read from /var/run/secrets/kubernetes.io/serviceaccount/token.
    The token is NEVER printed, logged, or included in repr() or exception messages.

Safety:
    - No oc exec.
    - No cluster mutations.
    - No Tekton.
    - kfp is optional; missing kfp raises a friendly RuntimeError.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
     docs/decisions/004-workbench-decorated-pipelines.md
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DSPA_SERVICE = "ds-pipeline-pipelines-definition"
_KFP_PORT = 8888

_DEFAULT_SA_TOKEN_PATH = pathlib.Path(
    "/var/run/secrets/kubernetes.io/serviceaccount/token"
)
_DEFAULT_SA_NAMESPACE_PATH = pathlib.Path(
    "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
)

# KFP v2 run terminal states (V2beta1RuntimeState string constants).
# These are the only states where polling should stop.
# Source: kfp_server_api.V2beta1RuntimeState.allowable_values
_TERMINAL_RUN_STATES: frozenset[str] = frozenset({
    "SUCCEEDED",
    "FAILED",
    "CANCELED",
    "SKIPPED",
})
_SUCCESS_RUN_STATES: frozenset[str] = frozenset({"SUCCEEDED"})


# ---------------------------------------------------------------------------
# _require_kfp_kubernetes — guard for optional kfp-kubernetes dependency
# ---------------------------------------------------------------------------


def _require_kfp_kubernetes(feature: str = "Kubernetes secret injection") -> object:
    """Return the kfp_kubernetes module, or raise a friendly ImportError if absent.

    This guard is ONLY called when a kfp-kubernetes feature is explicitly
    requested by the caller (e.g. secret injection, PVC mounting).
    It is never called at module import time.

    Args:
        feature: Human-readable name of the feature requiring kfp-kubernetes.
                 Included in the error message so callers get actionable output.

    Returns:
        The kfp_kubernetes module object.

    Raises:
        ImportError: If kfp_kubernetes is not importable, with install instructions.

    Usage (inside a function that uses kfp-kubernetes, not at module level)::

        def attach_secret(pipeline_task, secret_name: str) -> None:
            kfp_kubernetes = _require_kfp_kubernetes("secret injection")
            kfp_kubernetes.use_secret_as_env(pipeline_task, secret_name=secret_name)
    """
    try:
        import kfp_kubernetes  # noqa: PLC0415
        return kfp_kubernetes
    except ImportError as exc:
        raise ImportError(
            f"kfp-kubernetes is required for {feature}. "
            "Install it with:\n"
            "    pip install 'kfp-kubernetes>=1.2'\n"
            "The PRAGMA workbench image already includes kfp-kubernetes. "
            "See docs/openshift-image-contract.md."
        ) from exc


# ---------------------------------------------------------------------------
# get_dspa_endpoint
# ---------------------------------------------------------------------------


def get_dspa_endpoint(
    namespace: str | None = None,
    namespace_file: pathlib.Path | None = None,
) -> str:
    """Resolve the in-cluster KFP v2 API endpoint URL.

    Endpoint resolution order:
      1. PRAGMA_DSPA_ENDPOINT env var — returned verbatim (any URL accepted)
      2. namespace kwarg — explicit namespace argument bypasses all env vars
      3. PRAGMA_DSPA_NAMESPACE env var
      4. PRAGMA_TEST_NAMESPACE env var
      5. SA namespace file (default: /var/run/secrets/.../namespace)
      6. RuntimeError — no namespace source available

    Args:
        namespace:      Explicit namespace string. When provided, skips all
                        env var lookups and constructs the URL directly.
        namespace_file: Override for the SA namespace file path. Defaults to
                        /var/run/secrets/kubernetes.io/serviceaccount/namespace.

    Returns:
        https://ds-pipeline-pipelines-definition.<namespace>.svc.cluster.local:8888

    Raises:
        RuntimeError: When no namespace source is available and
                      PRAGMA_DSPA_ENDPOINT is not set.
    """
    # Priority 1: explicit endpoint env var — use verbatim.
    explicit_endpoint = os.environ.get("PRAGMA_DSPA_ENDPOINT")
    if explicit_endpoint:
        return explicit_endpoint

    # Resolve namespace from argument, then env vars, then SA namespace file.
    ns: str | None = namespace

    if ns is None:
        ns = os.environ.get("PRAGMA_DSPA_NAMESPACE") or None

    if ns is None:
        ns = os.environ.get("PRAGMA_TEST_NAMESPACE") or None

    if ns is None:
        ns_path = (
            namespace_file
            if namespace_file is not None
            else _DEFAULT_SA_NAMESPACE_PATH
        )
        if ns_path.exists():
            try:
                ns = ns_path.read_text().strip() or None
            except OSError:
                ns = None

    if not ns:
        raise RuntimeError(
            "Cannot resolve DSPA namespace. Set one of:\n"
            "  PRAGMA_DSPA_ENDPOINT=<full URL>    — explicit endpoint (any URL)\n"
            "  PRAGMA_DSPA_NAMESPACE=<namespace>  — namespace for in-cluster URL\n"
            "  PRAGMA_TEST_NAMESPACE=<namespace>  — fallback namespace\n"
            "Or pass namespace=<namespace> to get_dspa_endpoint().\n"
            "When running inside a pod the namespace is read from:\n"
            f"  {_DEFAULT_SA_NAMESPACE_PATH}"
        )

    return f"https://{_DSPA_SERVICE}.{ns}.svc.cluster.local:{_KFP_PORT}"


# ---------------------------------------------------------------------------
# get_service_account_token
# ---------------------------------------------------------------------------


def get_service_account_token(
    token_path: pathlib.Path | None = None,
) -> str | None:
    """Read the Kubernetes service account token from the pod mount.

    The token value is returned as a stripped string. It is NEVER printed,
    logged, or included in any repr or exception message anywhere in this
    module.

    Args:
        token_path: Override path to the SA token file. Defaults to the
                    standard pod mount at
                    /var/run/secrets/kubernetes.io/serviceaccount/token.

    Returns:
        Stripped token string if the file exists and is readable.
        None when running outside a Kubernetes pod (file absent).
        None on OSError (e.g. permission denied).
    """
    path = token_path if token_path is not None else _DEFAULT_SA_TOKEN_PATH

    if not path.exists():
        return None

    try:
        return path.read_text().strip() or None
    except OSError:
        return None


# ---------------------------------------------------------------------------
# DSPAConfig
# ---------------------------------------------------------------------------


class DSPAConfig:
    """Lightweight endpoint + auth configuration for DSPA/KFP v2.

    The token value is stored privately and is NEVER exposed via repr(),
    str(), or any public attribute. Use has_token to check whether a token
    was provided without revealing its value.

    Attributes:
        endpoint:  The KFP API endpoint URL.
        has_token: True if a non-None token was provided, False otherwise.

    Example::

        config = DSPAConfig(
            endpoint=get_dspa_endpoint(),
            token=get_service_account_token(),
        )
        client = make_kfp_client(config.endpoint, config._token)

    Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
    """

    def __init__(self, endpoint: str, token: str | None = None) -> None:
        self.endpoint: str = endpoint
        self.has_token: bool = token is not None
        # Token stored privately — not in repr(), str(), or __dict__ output.
        self._token: str | None = token

    def __repr__(self) -> str:
        return (
            f"DSPAConfig(endpoint={self.endpoint!r}, has_token={self.has_token})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# make_kfp_client
# ---------------------------------------------------------------------------


def make_kfp_client(endpoint: str, token: str | None = None) -> Any:
    """Construct a kfp.Client for the given DSPA endpoint.

    kfp is imported lazily so this module can be loaded without kfp installed.
    Token is passed as existing_token if provided. Token is NEVER printed.

    Args:
        endpoint: Full KFP API endpoint URL:
                    http://ds-pipeline-pipelines-definition.<ns>.svc.cluster.local:8888
        token:    Optional SA bearer token string. When provided, passed as
                  existing_token to kfp.Client. Never logged or printed.

    Returns:
        kfp.Client instance configured for the endpoint.

    Raises:
        RuntimeError: If kfp is not installed. Message includes 'kfp' and
                      'install' so the user knows how to resolve it.
    """
    try:
        import kfp  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "make_kfp_client() requires the kfp SDK to be installed.\n"
            "Install it with: pip install kfp\n"
            "The PRAGMA workbench notebook image includes kfp pre-installed."
        ) from exc

    if token is not None:
        return kfp.Client(host=endpoint, existing_token=token, verify_ssl=False)
    return kfp.Client(host=endpoint, verify_ssl=False)


# ---------------------------------------------------------------------------
# upload_pipeline
# ---------------------------------------------------------------------------


def upload_pipeline(
    client: Any,
    yaml_path: str | pathlib.Path,
    pipeline_name: str,
) -> str:
    """Upload a compiled KFP v2 pipeline YAML to the DSPA pipeline registry.

    Args:
        client:        kfp.Client instance from make_kfp_client().
        yaml_path:     Path to the compiled pipeline YAML. Must exist before
                       calling — use PragmaPipeline.compile(path) to produce it.
        pipeline_name: Name for the pipeline in the DSPA registry.

    Returns:
        pipeline_id string from the DSPA response. Use this ID to create runs.

    Raises:
        FileNotFoundError: If yaml_path does not exist. Catches missing-compile
                           step early, before attempting any network call.
    """
    path = pathlib.Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline YAML not found: {path}\n"
            "Run PragmaPipeline.compile(path) before upload_pipeline().\n"
            "Example:\n"
            "  run.compile('pipeline/generated/pragma-s.yaml')\n"
            "  pipeline_id = upload_pipeline("
            "client, 'pipeline/generated/pragma-s.yaml', 'pragma-s')"
        )

    response = client.upload_pipeline(str(path), pipeline_name=pipeline_name)
    return str(response.pipeline_id)


# ---------------------------------------------------------------------------
# submit_pipeline_run
# ---------------------------------------------------------------------------


def submit_pipeline_run(
    client: Any,
    pipeline_id: str,
    run_name: str,
    arguments: dict[str, Any] | None = None,
    experiment_name: str | None = None,
) -> str:
    """Create a KFP pipeline run on the DSPA.

    Args:
        client:          kfp.Client from make_kfp_client().
        pipeline_id:     Pipeline ID returned by upload_pipeline().
        run_name:        Name for this specific run instance.
        arguments:       Optional dict of pipeline parameter overrides,
                         e.g. {"model_size": "S", "max_steps": 1}.
        experiment_name: Optional KFP experiment name. The experiment is
                         created if it does not already exist.

    Returns:
        run_id string from the DSPA response. Use this to track the run
        via the OpenShift AI dashboard or the KFP v2 API.
    """
    # Resolve version_id — kfp v2 run_pipeline() requires both pipeline_id and version_id.
    versions = client.list_pipeline_versions(pipeline_id=pipeline_id)
    version_id = versions.pipeline_versions[0].pipeline_version_id

    # Resolve or create experiment — run_pipeline() requires an experiment_id.
    name = experiment_name or "Default"
    try:
        exp = client.get_experiment(experiment_name=name)
    except Exception:  # noqa: BLE001
        exp = client.create_experiment(name=name)
    experiment_id = exp.experiment_id

    response = client.run_pipeline(
        experiment_id=experiment_id,
        job_name=run_name,
        pipeline_id=pipeline_id,
        version_id=version_id,
        params=arguments,
    )
    return str(response.run_id)


# ---------------------------------------------------------------------------
# get_run_status
# ---------------------------------------------------------------------------


def get_run_status(client: Any, run_id: str) -> str:
    """Return the current state of a KFP v2 run as an uppercase string.

    Wraps kfp.Client.get_run() and normalises the state to a plain uppercase
    string regardless of whether the SDK returns a string constant or an
    enum-like object.

    Args:
        client: kfp.Client from make_kfp_client().
        run_id: Run ID returned by submit_pipeline_run().

    Returns:
        Uppercase state string, e.g. "PENDING", "RUNNING", "SUCCEEDED",
        "FAILED", "CANCELED", "SKIPPED".
        Returns "UNKNOWN" if the state cannot be determined.
    """
    run = client.get_run(run_id=run_id)
    state = getattr(run, "state", None)
    if state is None:
        return "UNKNOWN"
    # state is a plain string constant in kfp 2.7.0 (V2beta1RuntimeState).
    # Handle enum-like objects defensively for forward compatibility.
    if hasattr(state, "value"):
        return str(state.value).upper()
    return str(state).upper()


# ---------------------------------------------------------------------------
# wait_for_run_terminal
# ---------------------------------------------------------------------------


def wait_for_run_terminal(
    client: Any,
    run_id: str,
    timeout: int = 300,
    poll_interval: int = 10,
) -> str:
    """Poll a KFP v2 run until it reaches a terminal state or timeout.

    Terminal states: SUCCEEDED, FAILED, CANCELED, SKIPPED.
    Non-terminal states (PENDING, RUNNING, CANCELING, PAUSED,
    RUNTIME_STATE_UNSPECIFIED) are polled every poll_interval seconds.

    The token value is never accessed or printed by this function.

    Args:
        client:        kfp.Client from make_kfp_client().
        run_id:        Run ID returned by submit_pipeline_run().
        timeout:       Maximum seconds to wait before raising TimeoutError.
                       Default: 300. Set PRAGMA_TEST_TIMEOUT_SECONDS in the
                       environment to override in cluster tests.
        poll_interval: Seconds between status polls. Default: 10.

    Returns:
        Terminal state string, e.g. "SUCCEEDED", "FAILED".

    Raises:
        TimeoutError: If the run has not reached a terminal state within
                      the timeout period. The message includes the last
                      observed state and a diagnostic hint.
    """
    deadline = time.monotonic() + timeout
    while True:
        state = get_run_status(client=client, run_id=run_id)
        if state in _TERMINAL_RUN_STATES:
            return state
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Run {run_id!r} did not reach a terminal state within {timeout}s. "
                f"Last observed state: {state!r}. "
                "Check the OpenShift AI dashboard or:\n"
                f"  oc get pods -n <namespace> -l pipeline/runid={run_id}"
            )
        time.sleep(min(poll_interval, max(0.1, remaining)))
