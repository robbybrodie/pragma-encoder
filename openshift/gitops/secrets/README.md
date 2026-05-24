# openshift/gitops/secrets/

ArgoCD-synced Sealed Secrets for the `pragma-encoder` namespace.

---

## Why these files are safe to commit

Both files use [Bitnami Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets).
The credential data in `spec.encryptedData` is asymmetrically encrypted — only the
cluster's Sealed Secrets controller (private key stored in `kube-system`, never committed
or exported) can decrypt it. Committing the encrypted form is the intended and safe workflow.

The `spec.template.metadata` sections are **plain YAML** (not encrypted). They contain
Kubernetes resource metadata only — no credential values.

---

## Files

| File | Decrypts to | Contents |
|---|---|---|
| `registry-pull-secret.sealed.yaml` | Secret `pragma-registry` | NGC dockerconfigjson for nvcr.io pulls |
| `workbench-runtime-secret.sealed.yaml` | Secret `pragma-workbench-env` | AWS_* S3 credentials (RHOAI S3 Connection) |

---

## Connection ownership — workbench-runtime-secret

`workbench-runtime-secret.sealed.yaml` is the **authoritative source** of the
`pragma-workbench-env` Secret and the RHOAI S3 Data Connection:

```
Git (this file)
  └─► ArgoCD (wave 0 sync)
        └─► Sealed Secrets controller (decrypts in-cluster)
              └─► Secret pragma-workbench-env
                    └─► RHOAI dashboard (S3 Connection: "PRAGMA Object Storage")
```

The RHOAI dashboard recognises the Secret as a Connection because
`spec.template.metadata` carries the required annotations:

```yaml
spec:
  template:
    metadata:
      annotations:
        opendatahub.io/managed: "true"
        opendatahub.io/connection-type: s3
        opendatahub.io/display-name: "PRAGMA Object Storage"
      labels:
        opendatahub.io/dashboard: "true"
```

These annotations are plain YAML and can be edited without re-sealing the credentials.

---

## Credential keys

The workbench secret contains the native RHOAI S3 Connection schema
(source of truth: `oc get cm s3 -n redhat-ods-applications -o yaml`):

| Key | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | S3 access key ID |
| `AWS_SECRET_ACCESS_KEY` | S3 secret access key |
| `AWS_S3_ENDPOINT` | S3 endpoint URL |
| `AWS_DEFAULT_REGION` | S3 region |
| `AWS_S3_BUCKET` | S3 bucket name |

Non-S3 credentials (NGC API key, HuggingFace token) are **not** in this secret.
They belong in `registry-pull-secret.sealed.yaml` or a separate registry secret.

---

## Re-sealing after credential rotation

```bash
# 1. Copy the template (gitignored output)
cp openshift/secrets/workbench-secret.template.yaml \
   openshift/secrets/workbench-secret.yaml

# 2. Fill in real values in workbench-secret.yaml (never commit this file)

# 3. Seal with kubeseal (must be connected to the target cluster)
kubeseal --scope namespace-wide \
  --namespace pragma-encoder \
  --format yaml \
  < openshift/secrets/workbench-secret.yaml \
  > openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml

# 4. Verify the sealed file looks correct (spec.encryptedData has all keys)
# 5. Delete the populated workbench-secret.yaml
# 6. Commit the sealed file
```

The same procedure applies to `registry-pull-secret.sealed.yaml` using
`openshift/secrets/registry-credentials.template.yaml` as the source template.

---

## Never commit plaintext secrets

`.gitignore` excludes `openshift/secrets/*.yaml` (any populated secret file).
Only `.template.yaml` files and `.sealed.yaml` files are committed.

See also: `docs/deployment.md`, `docs/openshift-storage-pattern.md`
