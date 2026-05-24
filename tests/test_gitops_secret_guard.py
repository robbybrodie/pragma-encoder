"""Plaintext secret guard tests for GitOps and documentation paths.

Scans committed files for obvious credential leakage. These tests run
statically — no cluster, no network, no S3 required.

What is checked:
  - openshift/gitops/secrets/*.sealed.yaml: only SealedSecrets (encrypted), no raw Secret
  - openshift/secrets/: only .template.yaml files committed (checked via git ls-files)
  - tests/openshift/fixtures/: no real credential values in committed YAML
  - docs/: no real credential values in markdown
  - openshift/gitops/ tree: no raw kind: Secret

Patterns that trigger a failure:
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / etc. followed by a non-empty,
  non-placeholder, non-kubeseal-ciphertext value.

What is intentionally NOT flagged:
  - Values of "REPLACE_ME" (template placeholder)
  - Empty values ("")
  - Values starting with "Ag" + 20+ chars (kubeseal ciphertext)
  - Inline YAML comments (stripped before checking)

What is NOT checked (by design):
  - spec.encryptedData in SealedSecrets — legitimate ciphertext payload.

Reference: docs/deployment.md §Secrets, openshift/gitops/secrets/README.md
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import yaml

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent

_GITOPS_SECRETS = _REPO_ROOT / "openshift" / "gitops" / "secrets"
_OPENSHIFT_SECRETS = _REPO_ROOT / "openshift" / "secrets"
_FIXTURES = _REPO_ROOT / "tests" / "openshift" / "fixtures"
_DOCS = _REPO_ROOT / "docs"

# ---------------------------------------------------------------------------
# Credential key names to check for plaintext leakage
# ---------------------------------------------------------------------------

_CREDENTIAL_KEYS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_ENDPOINT",
    "AWS_S3_BUCKET",
    "AWS_DEFAULT_REGION",
    "NGC_API_KEY",
    "HF_TOKEN",
    "AI_PLATFORM_API_KEY",
]

# Matches a YAML key: <value> line (value may be followed by a comment)
_KEY_VALUE_RE = {
    key: re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*)$")
    for key in _CREDENTIAL_KEYS
}


def _extract_yaml_value(line: str, key: str) -> str | None:
    """Extract the YAML value for ``key`` from ``line``, stripping inline comments.

    Returns None if ``line`` does not start a ``key: value`` assignment.
    Returns the bare value string (empty string if value is absent).
    """
    m = _KEY_VALUE_RE[key].match(line)
    if m is None:
        return None
    raw = m.group(1)
    # Strip inline YAML comment: anything after unquoted ' #'
    # Simple heuristic sufficient for unquoted scalar values.
    raw = re.sub(r"\s+#.*$", "", raw).strip()
    return raw


def _is_plaintext_credential(value: str) -> bool:
    """Return True if ``value`` looks like a real credential (not a placeholder or ciphertext)."""
    if not value:
        return False
    if value == "REPLACE_ME" or value.startswith("REPLACE_ME"):
        return False
    # kubeseal ciphertext: starts with 'Ag' and is very long (>80 chars base64)
    if value.startswith("Ag") and len(value) > 80:
        return False
    # Template substitution markers: ${VAR}, <VAR>, <<VAR>>
    if value.startswith("${") or value.startswith("<"):
        return False
    return True


def _scan_for_plaintext_creds(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (line_number, line) pairs where a plaintext credential is suspected."""
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        for key in _CREDENTIAL_KEYS:
            value = _extract_yaml_value(line, key)
            if value is not None and _is_plaintext_credential(value):
                hits.append((i, line.strip()))
                break  # only report each line once
    return hits


def _git_ls_files(*paths: str) -> list[pathlib.Path]:
    """Return the list of files tracked by Git under the given paths."""
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "ls-files", "--", *paths],
        capture_output=True,
        text=True,
    )
    return [
        _REPO_ROOT / f
        for f in result.stdout.splitlines()
        if f
    ]


# ---------------------------------------------------------------------------
# 1. sealed.yaml files must be SealedSecrets, not raw Secrets
# ---------------------------------------------------------------------------


class TestSealedSecretFiles:
    """Files in openshift/gitops/secrets/*.sealed.yaml must be SealedSecrets.

    A raw Secret committed here would expose credentials in plaintext
    (base64-encoded, which is not encryption).
    """

    def _sealed_files(self) -> list[pathlib.Path]:
        return sorted(_GITOPS_SECRETS.glob("*.sealed.yaml"))

    def test_sealed_secret_files_exist(self) -> None:
        """At least one .sealed.yaml must exist (sanity check)."""
        files = self._sealed_files()
        assert files, (
            "No *.sealed.yaml files found in openshift/gitops/secrets/. "
            "Expected at least workbench-runtime-secret.sealed.yaml."
        )

    def test_all_sealed_files_are_kind_sealedsecret(self) -> None:
        """Every *.sealed.yaml must be kind: SealedSecret, never kind: Secret."""
        for sealed_file in self._sealed_files():
            try:
                doc = yaml.safe_load(sealed_file.read_text())
            except yaml.YAMLError as exc:
                raise AssertionError(
                    f"{sealed_file.relative_to(_REPO_ROOT)}: YAML parse error: {exc}"
                ) from exc
            kind = doc.get("kind") if isinstance(doc, dict) else None
            assert kind == "SealedSecret", (
                f"{sealed_file.relative_to(_REPO_ROOT)}: kind must be 'SealedSecret'. "
                f"Found: {kind!r}. "
                "A raw Secret committed to Git would expose credentials in base64. "
                "Use 'kubeseal' to encrypt before committing."
            )

    def test_sealed_files_have_no_string_data_section(self) -> None:
        """SealedSecrets must not have a stringData section (would be plaintext)."""
        for sealed_file in self._sealed_files():
            try:
                doc = yaml.safe_load(sealed_file.read_text())
            except yaml.YAMLError:
                continue  # parse error caught by test above
            if not isinstance(doc, dict):
                continue
            assert "stringData" not in doc, (
                f"{sealed_file.relative_to(_REPO_ROOT)}: contains 'stringData'. "
                "SealedSecrets must only have 'spec.encryptedData'. "
                "A stringData section would expose credentials in plaintext."
            )

    def test_sealed_files_have_no_plaintext_cred_lines(self) -> None:
        """No *.sealed.yaml file may contain a plaintext credential value."""
        for sealed_file in self._sealed_files():
            hits = _scan_for_plaintext_creds(sealed_file)
            assert not hits, (
                f"{sealed_file.relative_to(_REPO_ROOT)}: suspected plaintext credential(s):\n"
                + "\n".join(f"  line {ln}: {line}" for ln, line in hits)
                + "\nRe-seal with kubeseal before committing. "
                "Encrypted values start with 'Ag...' (kubeseal ciphertext prefix)."
            )


# ---------------------------------------------------------------------------
# 2. openshift/secrets/ — only template files committed to Git
# ---------------------------------------------------------------------------


class TestSecretsDirectoryTemplatesOnly:
    """openshift/secrets/ must only have *.template.yaml files tracked by Git.

    Populated secret files are gitignored and allowed to exist locally
    for sealing workflows. This test checks Git-tracked files only —
    locally-created gitignored files do not fail this test.
    """

    def test_no_non_template_yaml_committed_to_secrets_dir(self) -> None:
        """No *.yaml (non-template) file under openshift/secrets/ may be Git-tracked."""
        tracked = _git_ls_files("openshift/secrets/")
        non_template = [
            f for f in tracked
            if f.suffix == ".yaml" and not f.name.endswith(".template.yaml")
        ]
        assert not non_template, (
            "Populated secret files committed to openshift/secrets/:\n"
            + "\n".join(f"  {f.relative_to(_REPO_ROOT)}" for f in non_template)
            + "\nThese files must be gitignored (openshift/secrets/*.yaml). "
            "Only *.template.yaml files should be committed. "
            "Remove from Git: git rm --cached <file>"
        )

    def test_template_files_use_replace_me_placeholders(self) -> None:
        """*.template.yaml files must use REPLACE_ME, not real credential values."""
        if not _OPENSHIFT_SECRETS.exists():
            return

        for tmpl in sorted(_OPENSHIFT_SECRETS.glob("*.template.yaml")):
            hits = _scan_for_plaintext_creds(tmpl)
            assert not hits, (
                f"{tmpl.relative_to(_REPO_ROOT)}: real credential value(s) in template:\n"
                + "\n".join(f"  line {ln}: {line}" for ln, line in hits)
                + "\nTemplate files must use REPLACE_ME as placeholder values. "
                "Never commit real credentials, even in template files."
            )


# ---------------------------------------------------------------------------
# 3. tests/openshift/fixtures/ — no plaintext credentials in YAML fixtures
# ---------------------------------------------------------------------------


class TestFixtureFiles:
    """Test fixture YAML files must not contain plaintext credentials."""

    def test_fixture_yaml_files_have_no_plaintext_creds(self) -> None:
        """No YAML file under tests/openshift/fixtures/ may have plaintext creds."""
        if not _FIXTURES.exists():
            return

        for yaml_file in sorted(_FIXTURES.rglob("*.yaml")):
            hits = _scan_for_plaintext_creds(yaml_file)
            assert not hits, (
                f"{yaml_file.relative_to(_REPO_ROOT)}: suspected plaintext credential(s):\n"
                + "\n".join(f"  line {ln}: {line}" for ln, line in hits)
                + "\nFixture files must use REPLACE_ME or synthetic placeholder values. "
                "Never commit real credentials in test fixtures."
            )

    def test_fixture_yaml_templates_use_replace_me(self) -> None:
        """*.yaml.template files must use REPLACE_ME placeholders for credentials."""
        if not _FIXTURES.exists():
            return

        for tmpl_file in sorted(_FIXTURES.rglob("*.yaml.template")):
            hits = _scan_for_plaintext_creds(tmpl_file)
            assert not hits, (
                f"{tmpl_file.relative_to(_REPO_ROOT)}: suspected real credentials in template:\n"
                + "\n".join(f"  line {ln}: {line}" for ln, line in hits)
                + "\nTemplate files must use REPLACE_ME as placeholder values."
            )


# ---------------------------------------------------------------------------
# 4. docs/ — no plaintext credentials in markdown files
# ---------------------------------------------------------------------------


class TestDocumentationFiles:
    """Documentation files must not contain plaintext credentials."""

    def test_docs_markdown_has_no_plaintext_creds(self) -> None:
        """No .md file under docs/ may contain plaintext credential values."""
        if not _DOCS.exists():
            return

        for md_file in sorted(_DOCS.rglob("*.md")):
            hits = _scan_for_plaintext_creds(md_file)
            assert not hits, (
                f"{md_file.relative_to(_REPO_ROOT)}: suspected plaintext credential(s):\n"
                + "\n".join(f"  line {ln}: {line}" for ln, line in hits)
                + "\nDocumentation must use REPLACE_ME or synthetic placeholder values. "
                "Never embed real credentials in docs."
            )


# ---------------------------------------------------------------------------
# 5. openshift/gitops/ — no raw Secret kind (belt-and-suspenders)
# ---------------------------------------------------------------------------


class TestGitopsNoRawSecrets:
    """No file under openshift/gitops/ may be kind: Secret.

    All secrets managed by ArgoCD must be SealedSecrets. A raw Secret
    committed here would expose credentials in base64 (not encryption).
    """

    def test_no_kind_secret_in_gitops_tree(self) -> None:
        """No YAML under openshift/gitops/ may be kind: Secret."""
        gitops_dir = _REPO_ROOT / "openshift" / "gitops"
        if not gitops_dir.exists():
            return

        raw_secrets = []
        for yaml_file in sorted(gitops_dir.rglob("*.yaml")):
            try:
                doc = yaml.safe_load(yaml_file.read_text())
            except yaml.YAMLError:
                continue
            if isinstance(doc, dict) and doc.get("kind") == "Secret":
                raw_secrets.append(yaml_file)

        assert not raw_secrets, (
            "Raw Secret files detected in openshift/gitops/:\n"
            + "\n".join(f"  {f.relative_to(_REPO_ROOT)}" for f in raw_secrets)
            + "\nAll secrets in openshift/gitops/ must be kind: SealedSecret. "
            "Use kubeseal to encrypt before committing."
        )
