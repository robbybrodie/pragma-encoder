#!/usr/bin/env bash
# bootstrap-project.sh
# Bootstraps the pragma-encoder project on OpenShift AI.
#
# Usage:
#   ./openshift/scripts/bootstrap-project.sh
#
# Prerequisites:
#   - oc CLI logged in with cluster-admin
#   - ArgoCD installed (openshift-gitops operator)
#   - Secrets populated in openshift/secrets/ (copy from templates)

set -euo pipefail

NAMESPACE="pragma-encoder"
ARGOCD_NAMESPACE="openshift-gitops"

echo "Bootstrapping pragma-encoder on OpenShift AI..."

# Create namespace
oc apply -f openshift/gitops/namespace.yaml

# Apply Sealed Secrets (if using sealed-secrets controller)
# oc apply -f openshift/gitops/secrets/

# Apply RBAC
oc apply -f openshift/gitops/rbac/service-accounts.yaml

# Register ArgoCD application
oc apply -f openshift/argocd/application.yaml

echo "ArgoCD application registered. Sync will start automatically."
echo "Monitor at: oc get application pragma-encoder -n $ARGOCD_NAMESPACE"
echo ""
echo "To trigger a manual sync:"
echo "  argocd app sync pragma-encoder"
