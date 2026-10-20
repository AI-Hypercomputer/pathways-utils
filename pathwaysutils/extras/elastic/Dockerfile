# Workload image for the Pathways elastic training demo.
#
# Build from the repository root so this checkout of pathwaysutils is
# installed in the image (gke_demo_bridge.py --build-image does this for you):
#
#   docker build -f pathwaysutils/extras/elastic/Dockerfile -t IMAGE .
#
# The head pod runs with JAX_PLATFORMS=proxy, so a CPU-only JAX is enough; the
# TPUs are driven by the Pathways server containers.
FROM python:3.12-slim

ARG ORBAX_REF=main

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app/pathways-utils

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "/app/pathways-utils[elastic-demo]" \
    && pip install --no-cache-dir --no-deps \
         "git+https://github.com/google/orbax.git@${ORBAX_REF}#subdirectory=checkpoint"

CMD ["python3", "-m", "pathwaysutils.extras.elastic.elastic_workload", "--help"]
