"""Safe `oc` wrapper for PRAGMA OpenShift integration tests.

Provides a thin, safety-enforcing wrapper around the `oc` CLI.

Safety contract:
- All delete commands are validated before execution.
- Deletes are rejected unless label-scoped to test-owned resources.
- Namespace/Secret/ServiceAccount deletes are unconditionally rejected.
- Secret data is never printed — redact() strips obvious sensitive values.
- No destructive broad commands are permitted.

Usage::

    from tests.openshift.oc import oc, oc_json, resource_exists, crd_exists

    oc(["get", "pods"], namespace="pragma-encoder")
    result = oc_json(["get", "pod", "my-pod"], namespace="pragma-encoder")
    if resource_exists("serviceaccount", "pragma-encoder-training", namespace="pragma-encoder"):
        ...
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any

# Label key:values that mark test-owned resources.
_TEST_RUN_LABEL = "pragma.redhat.com/test-run"
_TEST_ID_LABEL  = "pragma.redhat.com/test-id"
_TEST_RUN_VALUE = "true"

# Resource kinds that must never be deleted by tests.
_FORBIDDEN_DELETE_KINDS: frozenset[str] = frozenset({
    "namespace", "namespaces",
    "secret", "secrets",
    "serviceaccount", "serviceaccounts", "sa",
})

# Patterns used by redact() to mask obvious secret values.
_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Bearer token in Authorization header or env value
    (re.compile(r"(Bearer\s+)[A-Za-z0-9\-_.~+/]+=*", re.IGNORECASE), r"\1<redacted>"),
    # Generic token= or password= key-value (URL or shell form)
    (re.compile(r"(token|password|passwd|secret|key|credential)=[^\s&,;\"']+",
                re.IGNORECASE), r"\1=<redacted>"),
    # Kubernetes JWT-style tokens (long base64 dot-separated strings)
    (re.compile(r"eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+"),
     "<redacted-jwt>"),
]


def redact(text: str) -> str:
    """Redact obvious secret/token/password values from a string.

    Applies basic regex substitutions to prevent secrets from appearing in
    test failure messages or logs. Not exhaustive — this is a safety net,
    not a comprehensive secret scanner.

    Args:
        text: Raw string that may contain sensitive values.

    Returns:
        String with obvious sensitive values replaced by ``<redacted>``.
    """
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def require_oc() -> None:
    """Assert that the ``oc`` binary is present on PATH.

    Raises:
        AssertionError: If ``oc`` is not found on PATH.
    """
    assert shutil.which("oc") is not None, (
        "oc binary not found on PATH. "
        "Install the OpenShift CLI: https://docs.openshift.com/container-platform/latest/cli_reference/openshift_cli/getting-started-cli.html"
    )


def validate_safe_delete(args: list[str]) -> None:
    """Reject delete commands that are not safely label-scoped to test resources.

    Called automatically by :func:`oc` before any delete is executed.
    Raises ``ValueError`` for:

    - Deleting forbidden resource kinds (namespace, secret, serviceaccount, sa).
    - Using ``--all`` flag.
    - Deleting without a ``-l``/``--selector`` that includes the test-run label.

    Non-delete commands pass through without inspection.

    Args:
        args: Argument list (without the leading ``oc``).

    Raises:
        ValueError: If the delete is unsafe.
    """
    if not args or args[0] != "delete":
        return  # not a delete command — always safe to pass through

    args_str = " ".join(args)

    # Reject --all
    if "--all" in args:
        raise ValueError(
            "oc delete --all is forbidden in integration tests. "
            "Use label-scoped deletes only: "
            f"-l {_TEST_RUN_LABEL}={_TEST_RUN_VALUE}"
        )

    # Reject forbidden resource kinds
    # Argument list format: ["delete", "<kind>", "<name>", ...]
    # or ["delete", "<kind>/<name>", ...]
    # or ["delete", "-l", "<selector>", "<kind>", ...]
    # We scan args[1:] for any token that looks like a kind name.
    for token in args[1:]:
        # Strip slash-suffixed resource/name form (e.g. "secret/my-secret")
        kind = token.split("/")[0].lower().lstrip("-")
        if kind in _FORBIDDEN_DELETE_KINDS:
            raise ValueError(
                f"oc delete {token!r} is forbidden in integration tests. "
                f"Tests must never delete {kind} resources."
            )

    # Require a label selector containing the test-run label.
    has_label_selector = False
    for i, arg in enumerate(args):
        if arg in ("-l", "--selector"):
            selector_val = args[i + 1] if i + 1 < len(args) else ""
            if _TEST_RUN_LABEL in selector_val:
                has_label_selector = True
                break
        elif arg.startswith("-l") and _TEST_RUN_LABEL in arg:
            has_label_selector = True
            break
        elif arg.startswith("--selector=") and _TEST_RUN_LABEL in arg:
            has_label_selector = True
            break

    if not has_label_selector:
        raise ValueError(
            f"oc delete must include a label selector containing "
            f"'{_TEST_RUN_LABEL}={_TEST_RUN_VALUE}'. "
            f"Command: oc {args_str}. "
            "Tests may only delete their own labelled resources."
        )


def oc(
    args: list[str],
    namespace: str | None = None,
    check: bool = True,
    timeout: int | None = 30,
) -> subprocess.CompletedProcess[str]:
    """Run an ``oc`` command safely and return the completed process.

    Builds the command as ``["oc"] + args`` (with optional ``-n <namespace>``),
    captures stdout/stderr, and validates any delete command before execution.

    On failure (non-zero exit with ``check=True``), raises ``subprocess.CalledProcessError``
    with redacted stdout/stderr to avoid leaking secrets.

    Args:
        args:      Argument list, e.g. ``["get", "pods"]``.
        namespace: Namespace flag. Added as ``-n <namespace>`` after ``oc`` if provided.
        check:     Raise on non-zero exit code (default True).
        timeout:   Subprocess timeout in seconds (default 30). None = no timeout.

    Returns:
        :class:`subprocess.CompletedProcess` with captured stdout/stderr.

    Raises:
        AssertionError:               If ``oc`` binary is not on PATH.
        ValueError:                   If the command is an unsafe delete.
        subprocess.CalledProcessError: If the command exits non-zero and ``check=True``.
        subprocess.TimeoutExpired:    If the command exceeds ``timeout``.
    """
    require_oc()
    validate_safe_delete(args)

    cmd: list[str] = ["oc"]
    if namespace is not None:
        cmd += ["-n", namespace]
    cmd += args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,  # we check manually to redact output
        )
    except subprocess.TimeoutExpired as exc:
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=timeout,
            output=redact(exc.output or ""),
            stderr=redact(exc.stderr or ""),
        ) from exc

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=cmd,
            output=redact(result.stdout),
            stderr=redact(result.stderr),
        )

    return result


def oc_json(
    args: list[str],
    namespace: str | None = None,
    timeout: int | None = 30,
) -> dict[str, Any]:
    """Run an ``oc`` command and parse stdout as JSON.

    Appends ``-o json`` if not already present in ``args``.

    Args:
        args:      Argument list, e.g. ``["get", "pod", "my-pod"]``.
        namespace: Namespace flag passed to :func:`oc`.
        timeout:   Subprocess timeout in seconds (default 30).

    Returns:
        Parsed JSON as a Python dict.

    Raises:
        ValueError: If stdout is not valid JSON.
    """
    if "-o" not in args and "--output" not in args:
        args = list(args) + ["-o", "json"]

    result = oc(args, namespace=namespace, check=True, timeout=timeout)
    try:
        return json.loads(result.stdout)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"oc output was not valid JSON: {redact(result.stdout[:200])}"
        ) from exc


def resource_exists(
    kind: str,
    name: str,
    namespace: str | None = None,
) -> bool:
    """Return True if the named resource exists, False if it does not.

    Uses ``oc get <kind> <name>`` with a short timeout.
    Does not print or expose resource data.

    Args:
        kind:      Resource kind, e.g. ``"serviceaccount"``, ``"secret"``.
        name:      Resource name.
        namespace: Namespace to query. None = cluster-scoped resource.

    Returns:
        True if the resource exists, False otherwise.
    """
    result = oc(
        ["get", kind, name],
        namespace=namespace,
        check=False,
        timeout=15,
    )
    return result.returncode == 0


def crd_exists(name: str) -> bool:
    """Return True if a CustomResourceDefinition with the given name exists.

    Args:
        name: Full CRD name, e.g. ``"pipelineruns.tekton.dev"``.

    Returns:
        True if the CRD exists, False otherwise.
    """
    return resource_exists("crd", name, namespace=None)


def label_selector(test_id: str | None = None) -> str:
    """Return a label selector string scoped to test-owned resources.

    If ``test_id`` is provided, the selector is scoped to a specific test run.
    If not, the selector matches all resources created by any test run.

    Args:
        test_id: Optional unique test-run identifier.

    Returns:
        Comma-separated label selector string, e.g.
        ``"pragma.redhat.com/test-run=true,pragma.redhat.com/test-id=pragma-it-20260519-abc123"``
    """
    selector = f"{_TEST_RUN_LABEL}={_TEST_RUN_VALUE}"
    if test_id is not None:
        selector += f",{_TEST_ID_LABEL}={test_id}"
    return selector
