#!/usr/bin/env bash
# bootstrap-project.sh
# Bootstraps the pragma-encoder GitOps project on OpenShift AI.
#
# This script must be run ONCE as cluster-admin before ArgoCD can take over.
# After bootstrap, all subsequent changes are managed via Git + ArgoCD.
#
# Usage (from repo root):
#   oc login <cluster-url>          # must be cluster-admin
#   ./openshift/scripts/bootstrap-project.sh
#
# Prerequisites:
#   - oc CLI logged in as cluster-admin (oc whoami should return kube:admin)
#   - OpenShift GitOps operator installed (openshift-gitops namespace exists)
#   - Bitnami Sealed Secrets controller running in kube-system
#   - Sealed secret files committed in openshift/gitops/secrets/
#
# What this script does:
#   1. Creates the pragma-encoder namespace
#   2. Applies ArgoCD RBAC bootstrap (admin + SealedSecret access)
#      — ArgoCD cannot sync wave-0 resources without this
#   3. Applies SealedSecrets so the controller decrypts them before ArgoCD syncs
#   4. Registers the ArgoCD Application CR
#      — ArgoCD auto-sync then drives all remaining resources (waves 0–4)
#
# After this script completes:
#   - ArgoCD will sync RBAC, ImageStream, BuildConfig, DSPA, and Notebook
#   - The BuildConfig fires automatically (ConfigChange trigger)
#   - The Notebook pod starts once the image build completes

set -euo pipefail

NAMESPACE="pragma-encoder"
ARGOCD_NAMESPACE="openshift-gitops"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=== pragma-encoder bootstrap ==="
echo "Cluster: $(oc whoami --show-server)"
echo "User:    $(oc whoami)"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Namespace
# ---------------------------------------------------------------------------
echo "[1/4] Creating namespace ${NAMESPACE}..."
oc apply -f "${REPO_ROOT}/openshift/gitops/namespace.yaml"

# ---------------------------------------------------------------------------
# Step 2: ArgoCD bootstrap RBAC
# Must exist before ArgoCD can sync anything into the namespace.
# ---------------------------------------------------------------------------
echo "[2/4] Applying ArgoCD bootstrap RBAC..."
oc apply -f "${REPO_ROOT}/openshift/gitops/rbac/argocd-admin.yaml"
oc apply -f "${REPO_ROOT}/openshift/gitops/rbac/argocd-sealedsecret-access.yaml"

# ---------------------------------------------------------------------------
# Step 3: Sealed Secrets
# Apply now so the controller decrypts them before ArgoCD's first sync.
# ArgoCD will adopt and maintain these on subsequent syncs.
# ---------------------------------------------------------------------------
echo "[3/4] Applying SealedSecrets..."
oc apply -f "${REPO_ROOT}/openshift/gitops/secrets/registry-pull-secret.sealed.yaml"
oc apply -f "${REPO_ROOT}/openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml"

echo "      Waiting 10s for Sealed Secrets controller to decrypt..."
sleep 10
oc get secret pragma-registry -n "${NAMESPACE}" > /dev/null 2>&1 \
  && echo "      pragma-registry: OK" \
  || echo "      WARNING: pragma-registry not yet decrypted — controller may be slow"
oc get secret pragma-workbench-env -n "${NAMESPACE}" > /dev/null 2>&1 \
  && echo "      pragma-workbench-env: OK" \
  || echo "      WARNING: pragma-workbench-env not yet decrypted — controller may be slow"

# ---------------------------------------------------------------------------
# Step 4: Register ArgoCD Application
# This triggers automated sync of all waves (namespace → RBAC → ImageStream
# → BuildConfig → DSPA → Notebook).
# ---------------------------------------------------------------------------
echo "[4/4] Registering ArgoCD Application..."
oc apply -f "${REPO_ROOT}/openshift/argocd/application.yaml"

echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "Monitor sync:"
echo "  oc get applications.argoproj.io pragma-encoder -n ${ARGOCD_NAMESPACE}"
echo ""
echo "ArgoCD UI:"
echo "  https://$(oc get route openshift-gitops-server -n ${ARGOCD_NAMESPACE} -o jsonpath='{.spec.host}' 2>/dev/null || echo '<route-not-found>')"
echo ""
echo "Namespace events:"
echo "  oc get events -n ${NAMESPACE} --sort-by=.lastTimestamp | tail -20"
echo ""
echo "Image build status (fires automatically via ConfigChange trigger):"
echo "  oc get builds -n ${NAMESPACE}"
echo ""
echo "Notebook pod (starts after image build completes):"
echo "  oc get pods -n ${NAMESPACE}"
