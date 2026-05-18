# Technical Debt Register

## TD-001: Shared S3 bucket with nemo-tfm deployment

**Date:** 2026-05-17
**Severity:** Low (demo environment)
**Status:** Accepted for demo, must resolve before production

### Description
The pragma-encoder DSPA shares an S3 bucket with the
nemo-tfm deployment (transaction-foundation-model-openshiftai).
Pipeline artifacts from both deployments are stored in the
same bucket, differentiated only by KFP-generated run ID prefixes.

### Risk
- No hard isolation between the two deployments
- A bucket policy change affecting one affects both
- Cost attribution is unclear across deployments
- In production, data governance requires separate buckets
  per workload with separate IAM policies

### Resolution
Create a dedicated S3 bucket for pragma-encoder:
  Bucket: pragma-encoder-pipelines-<unique-suffix>
  IAM policy: scoped to pragma-encoder namespace only
  Region: same as nemo-tfm bucket

Re-seal the workbench runtime secret with new bucket values:
  kubeseal --scope namespace-wide \
    --namespace pragma-encoder \
    --format yaml \
    < openshift/secrets/workbench-secret.yaml \
    > openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml

### References
- Adjacent repo bucket: see nemo-tfm DSPA configuration
- OpenShift AI DSPA documentation: docs.redhat.com

---

## TD-002: NGC API key shared across deployments

**Date:** 2026-05-17
**Severity:** Low (demo environment)
**Status:** Accepted for demo, must resolve before production

### Description
The NGC API key used for pulling nvcr.io/nvidia/nemo:25.09.01
is the same key used in the nemo-tfm deployment.

### Risk
- Key rotation in one deployment requires updating both
- Rate limits on NGC API key are shared
- No audit trail separation between deployments

### Resolution
Generate a dedicated NGC API key for pragma-encoder:
  Log in to ngc.nvidia.com
  Generate a new API key scoped to pragma-encoder usage
  Re-seal registry-pull-secret.sealed.yaml with new key
  Rotate the nemo-tfm key independently

---

## TD-003: Profile tokenizer not implemented

**Date:** 2026-05-18
**Severity:** Medium (blocks real training on profile data)
**Status:** Partially resolved — xa zero-padded as shortcut

### Description
EmbeddingAssembler accepts xa_key_ids, xa_val_ids, xa_pos_ids
for the profile state path (static customer attributes).
These are produced correctly by the assembler via Equation 1.

However no ProfileTokenizerPipeline exists to produce these
inputs from real static customer attributes (plan, region,
balance quantile, life-long events).

FinancialTokenizerPipeline handles event tokens only.

### Impact
In the smoke test and early training: synthetic integer
tensors in valid ID ranges exercise the xa path correctly.
In production: callers must supply xa inputs manually
until ProfileTokenizerPipeline is implemented.

### Partial resolution (2026-05-18)
PragmaDataset (src/data/pragma_dataset.py) implements the
xa zero-padding shortcut:
  - na = 1 (single minimal profile token)
  - xa_key_ids: filled with vocab_spec.key_start
  - xa_val_ids: filled with vocab_spec.value_start
  - xa_pos_ids: filled with 0
  - ta: filled with 0.0

This allows real TabFormer training to proceed without
a ProfileTokenizerPipeline.

### Full resolution
Implement src/tokenizer/profile_pipeline.py:
  ProfileTokenizerPipeline that tokenises static customer
  attributes and life-long events, returning xa_key_ids,
  xa_val_ids, xa_pos_ids, ta consistent with VocabularySpec.

### References
  Paper: Section 2.1.2 (profile state definition)
  Paper: Section 2.3.2 (ProfileStateEncoder inputs)
  EmbeddingAssembler: src/model/assembler.py

---

## TD-004: RHOAI native pipeline-config injection bypassed

**Date:** 2026-05-18
**Severity:** Low (demo environment)
**Status:** Accepted for GitOps deployment; operator path not viable without dashboard

### Description
The RHOAI notebook-controller webhook injects `KFP_API_HOST` into workbench
pods by reading a namespace Secret named `ds-pipeline-config`. In a healthy
RHOAI cluster, this secret is created by the RHOAI dashboard when a user
connects a pipelines server to a workbench through the UI.

In a GitOps/bootstrap deployment (workbench declared via ArgoCD, not the
dashboard), no dashboard interaction occurs and the secret is never created.
Manually creating `ds-pipeline-config` does not work: the webhook rejects it
with "Skipping mounting secret not managed by workbenches" regardless of
labels or ownerReferences, because it was not created through the dashboard's
ownership chain.

### Workaround
`openshift/gitops/workbench/notebook.yaml` declares `KFP_API_HOST` and
`KF_PIPELINES_ENDPOINT` explicitly as env vars pointing to the
namespace-local DSPA service (port 8888, direct HTTPS, no OAuth proxy):

  https://ds-pipeline-pipelines-definition.pragma-encoder.svc.cluster.local:8888

This is equivalent to the operator-injected value and survives pod restarts.
The cluster CA bundle (`/etc/pki/tls/custom-certs/ca-bundle.crt`, injected
by the Notebook Controller) covers the DSPA TLS certificate.

### Resolution
If a dashboard-created workbench is ever needed, delete the GitOps-managed
Notebook CR, recreate the workbench through the RHOAI dashboard (connecting
the pipelines server), export the resulting Notebook CR, and replace the
manifest in openshift/gitops/workbench/notebook.yaml.

---

## TD-005: PAD token used as UNK corruption in MaskingStrategy

**Date:** 2026-05-18
**Severity:** Low (training correctness; PAD and UNK are both excluded from MLM loss)
**Status:** Known deviation — documented, not yet resolved

### Description
`src/masking/strategy.py` uses `TokenizerPipeline.PAD_ID = 0` as the
corruption token for the "UNK replacement" path in masking (§2.3.5).

The paper specifies that a fraction of selected positions are replaced
with the [UNK] token as input dropout (excluded from MLM loss).

`TokenizerPipeline` defines four specials: PAD (0), MASK (1), CLS (2), SEP (3).
There is no dedicated [UNK] token. PAD_ID is used in its place.

```python
# strategy.py
_UNK_TOKEN_ID: int = TokenizerPipeline.PAD_ID   # 0 — [UNK] replacement (no global UNK)
```

### Risk
- `EmbeddingAssembler` may interpret ID=0 as a PAD token (valid embedding slot)
  rather than as an [UNK] corruption — the embedding value differs from intent
- If a future VocabularySpec version reserves ID=0 for a different purpose,
  this silent aliasing becomes a correctness bug
- The PAD embedding is shared between genuine padding positions and UNK-corrupted
  positions — the model cannot distinguish them during training

### Resolution
Add a dedicated [UNK] token to `TokenizerPipeline`'s special token set:
  `UNK_ID = 4` (appended after SEP)

Update `VocabularySpec` and `EmbeddingAssembler` to allocate an embedding
for the new UNK ID. Update `_UNK_TOKEN_ID` in `strategy.py` to use
`TokenizerPipeline.UNK_ID`.

Alternatively, accept PAD-as-UNK as a permanent simplification and update
the comment in `strategy.py` to remove the "TODO" framing.

### References
- Paper: Section 2.3.5 (masking corruption)
- Code: `src/masking/strategy.py` — `_UNK_TOKEN_ID` constant
- Code: `src/tokenizer/pipeline.py` — special token IDs
