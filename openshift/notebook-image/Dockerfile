# PRAGMA Encoder — OpenShift AI Notebook Image
#
# Extends the Red Hat OpenShift AI base notebook image with
# PRAGMA-specific dependencies.
#
# Build: oc start-build pragma-encoder -n pragma-encoder
# Or: docker build -t pragma-encoder:latest openshift/notebook-image/

FROM registry.redhat.io/ubi9/python-311:latest

USER root

# Install system dependencies
RUN dnf install -y git && dnf clean all

USER 1001

# Set working directory
WORKDIR /opt/app-root/src

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pragma-encoder source code
COPY --chown=1001:0 src/ ./src/
COPY --chown=1001:0 configs/ ./configs/
COPY --chown=1001:0 scripts/ ./scripts/
COPY --chown=1001:0 pipeline/ ./pipeline/
COPY --chown=1001:0 notebooks/ ./notebooks/
COPY --chown=1001:0 setup.py ./

# Install pragma-encoder as a package
RUN pip install --no-cache-dir -e .

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
