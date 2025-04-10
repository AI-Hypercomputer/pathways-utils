# Dockerfile for building the colocated python start wheel using a specific JAX base image
# Build using --build-arg JAX_VERSION=<version>, where <version> must be one of [0.5.1, 0.5.2, 0.5.3]
# Example: `docker build --build-arg JAX_VERSION=0.5.1 -t my-colocated-python-server .`

# --- Build Argument ---
# Defines the build argument for the JAX version.
# This MUST be declared before the first FROM instruction to be used within it.
ARG JAX_VERSION

# --- Base Image ---
# Use the specified JAX base image from Google Artifact Registry.
# The JAX_VERSION argument is substituted here.
FROM us-docker.pkg.dev/cloud-tpu-v2-images/pathways/colocated_python_server:jax-$JAX_VERSION

# --- Application Setup ---
# Set working directory
WORKDIR /app

# Copy the user's requirements file containing the dependencies to install
COPY requirements.txt .

# Install user dependencies, and check if they are compatible, if not fail
RUN . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# --- Runtime Configuration ---
# Set the default port (optional, might be set by base image)
ENV PORT 50051

# Command to run the application. Colocated Python is already installed in the base image.
CMD ["/bin/sh", "-c", ". venv/bin/activate && venv/bin/python3 main.py --port=50051"]