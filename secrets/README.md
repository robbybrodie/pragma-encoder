# Secrets — Team Instructions

This directory holds credential templates for the pragma-encoder project.
Populated credential files are gitignored. Only templates are committed.

---

## What credentials are needed

| Credential | Purpose | Template |
|---|---|---|
| OpenShift cluster credentials | `oc login`, ArgoCD | `../openshift/secrets/cluster-credentials.template.env` |
| NGC API Key | NVIDIA model registry access | `../openshift/secrets/ai-platform.template.env` |
| S3 / Object Storage | Training data, model artefacts | `../openshift/secrets/ai-platform.template.env` |
| Container registry | Pushing notebook images | `../openshift/secrets/registry-credentials.template.yaml` |
| Workbench environment | Injected into OpenShift AI notebook pod | `../openshift/secrets/workbench-secret.template.yaml` |

---

## How to populate credentials

1. Copy the template:
   ```bash
   cp openshift/secrets/cluster-credentials.template.env \
      openshift/secrets/cluster-credentials.env
   ```

2. Edit the copied file and replace `REPLACE_ME` values with real credentials.

3. Verify the populated file is gitignored:
   ```bash
   git status   # Should NOT show the populated file
   ```

4. For Kubernetes secrets: seal with `kubeseal` before committing:
   ```bash
   kubeseal --format yaml < openshift/secrets/workbench-secret.yaml \
       > openshift/gitops/secrets/sealed-workbench-secret.yaml
   ```
   The sealed file is safe to commit. The plaintext file must not be committed.

---

## How gitignore protects these files

The `.gitignore` blocks:
```
.env
.env.*
*secret*
*credential*
*token*
*apikey*
*.key
*.pem
secrets/
!secrets/README.md
!secrets/*.template.*
```

The pre-commit hook (``.githooks/pre-commit``) provides a second layer of
protection by scanning staged files and staged diff additions for:
- Sensitive filename patterns (`.env`, `kubeconfig`, `*secret*`, etc.)
- Secret content patterns (AWS keys, JWT tokens, GitHub PATs, NGC keys, etc.)

If the hook fires unexpectedly on a false positive:
```bash
git commit --no-verify   # Use sparingly; document reason in commit message
```

---

## Production secret management patterns

### Option 1: Sealed Secrets (recommended for OpenShift)
Encrypt secrets with `kubeseal` using the cluster's public key.
Sealed secrets are safe to commit to git.
The SealedSecrets controller decrypts them in the cluster.

```bash
# Install kubeseal CLI
brew install kubeseal

# Seal a secret
kubeseal --format yaml < my-secret.yaml > sealed-my-secret.yaml
```

### Option 2: External Secrets Operator (ESO)
Sync secrets from an external vault (HashiCorp Vault, AWS Secrets Manager,
Azure Key Vault) into Kubernetes secrets automatically.
No secrets stored in git at all.

### Option 3: OpenShift Secrets with RBAC
Store secrets directly in OpenShift with strict RBAC.
Only pipeline service accounts can read them.
Suitable for non-production environments.

---

## Files in this directory

```
README.md               — This file (committed)
*.template.*            — Credential templates (committed, safe)
*.env                   — Populated env files (gitignored, never commit)
*.yaml (populated)      — Populated YAML secrets (gitignored, never commit)
```
