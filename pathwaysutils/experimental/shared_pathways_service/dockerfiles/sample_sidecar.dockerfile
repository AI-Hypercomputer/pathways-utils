# Use the JAX image with the custom-built sidecar as the base.
FROM us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar:20260423-python_3.12-jax_0.10.0

# Set the working directory
WORKDIR /app

# 1. Upgrade pip and build tools
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --upgrade pip setuptools wheel

# 2. Clone MaxText
RUN git clone https://github.com/google/maxtext.git

# ADD THE CACHE MOUNT HERE
# Install the same version of JAX and JAXlib as the base image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r maxtext/src/dependencies/requirements/base_requirements/requirements.txt && \
    uv pip install --upgrade jax==0.10.0 jaxlib==0.10.0

# 3. (optional) Copy your local edits to MaxText requirements and src, if any.
# Make sure you're running this docker build from the root of your local MaxText
# checkout.
# COPY maxtext/src/dependencies/requirements/base_requirements/requirements.txt ./requirements.txt
# COPY maxtext/src /app/maxtext/src

# Ensure MaxText src is in PYTHONPATH
# ENV PYTHONPATH=/app/maxtext/src:$PYTHONPATH
