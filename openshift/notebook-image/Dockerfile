# PRAGMA Encoder — OpenShift AI Notebook Image
#
# Based on the RHOAI S2I generic data-science notebook image.
# The Kubeflow Notebook Controller (RHOAI) injects:
#   - USER / runAsUser (from namespace SCC)
#   - OAuth proxy sidecar container
#   - Jupyter port and probe configuration
#   - Service and Route
#
# Do NOT add USER, EXPOSE, or security context directives here.
#
# Source code is NOT baked into this image — the Notebook CR's initContainer
# clones https://github.com/robbybrodie/pragma-encoder.git at pod start.
# This image only provides the Python runtime environment.

FROM image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/s2i-generic-data-science-notebook:2024.2

# Copy and install PRAGMA-specific Python requirements.
# PyTorch, JupyterLab, and common data-science libraries are already in the base image.
# The S2I base image runs the build as UID 1001 — do NOT chown /opt/app-root here;
# the base image already sets group-writable permissions for OpenShift's random UID.
COPY requirements.txt /opt/app-root/src/requirements.txt
RUN pip install --no-cache-dir -r /opt/app-root/src/requirements.txt

# Point HOME at the PVC-backed mount so Jupyter can create its runtime dir.
ENV HOME=/opt/app-root/src

# The Kubeflow Notebook Controller (RHOAI) expects the container to start
# JupyterLab on port 8888. Using sh -c so NB_PREFIX is expanded at runtime.
# NB_PREFIX is injected by the Notebook Controller (e.g. /notebook/pragma-encoder/pragma-encoder-workbench).
CMD ["sh", "-c", "exec jupyter lab --port=8888 --ip=0.0.0.0 --no-browser --allow-root --ServerApp.allow_remote_access=True --ServerApp.token= --ServerApp.password= --ServerApp.base_url=${NB_PREFIX:-/} --ServerApp.root_dir=/opt/app-root/src"]
