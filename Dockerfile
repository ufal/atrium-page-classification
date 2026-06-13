# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ARG ATRIUM_RUNNER_IMAGE=""
ARG ATRIUM_RUNNER_REPO="https://github.com/ufal/atrium-page-classification"
ARG ATRIUM_RUNNER_REF=""
ENV ATRIUM_RUNNER_IMAGE=${ATRIUM_RUNNER_IMAGE} \
    ATRIUM_RUNNER_REPO=${ATRIUM_RUNNER_REPO} \
    ATRIUM_RUNNER_REF=${ATRIUM_RUNNER_REF} \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ libgl1 libglib2.0-0 poppler-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# GPU (CUDA 12.6) torch by default — CPU-only is not supported for inference.
# To build a CPU image explicitly pass --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
# Pin exact versions to match the local environment (torch 2.7.1+cu126,
# torchvision 0.22.1).  Installed from the CUDA index FIRST so the later
# `pip install -r requirements.txt` sees torch/torchvision already satisfied
# and does not pull a CPU-only wheel from PyPI.
#
# NOTE: `transformers<5` is intentionally NOT pinned here — it is pinned in
# requirements.txt (and service/requirements.txt). transformers 5.x constructs
# models on the `meta` device, which crashes the timm builders for the RegNetY
# (v4.3) and EfficientNetV2 (v1.3) checkpoints used by --best; the <5 pin is the
# meta-device fix and must stay consistent across every requirements file.
RUN pip install --index-url ${TORCH_INDEX_URL} torch==2.7.1 torchvision==0.22.1

COPY requirements.txt requirements-test.txt ./
COPY service/requirements.txt ./service-requirements.txt
RUN pip install -r requirements.txt -r service-requirements.txt -r requirements-test.txt

COPY . .

# gpu_check.py aborts with a clear message if no CUDA device is visible
RUN printf '%s\n' \
    'import sys, torch' \
    'if not torch.cuda.is_available():' \
    '    print("ERROR: No CUDA GPU detected. This image requires a GPU.", file=sys.stderr)' \
    '    print("Use docker-compose.gpu.yml overlay and ensure NVIDIA Container Toolkit is installed.", file=sys.stderr)' \
    '    sys.exit(1)' \
    > /app/gpu_check.py

RUN useradd --create-home --uid 10001 atrium \
    && mkdir -p /cache/huggingface /data /app/model /app/result \
    && chown -R atrium:atrium /app /cache /data
USER atrium

# Default: single-model directory inference (v4.3).  Args are passed straight
# through to run.py, so the memory-aware ensemble engine is available too, e.g.:
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
#       run --rm classify --dir --inner --best --parallel
# The GPU profile registry (model/gpu_profile.json) is written into the
# page-model volume and is hardware/torch/batch-keyed, so it is reused across
# runs on the same GPU and auto-invalidated (re-profiled) on a different one.
ENTRYPOINT ["sh", "-c", "python3 /app/gpu_check.py && exec python3 run.py \"$@\"", "--"]
CMD ["-d", "/data/input", "--hf", "-rev", "v4.3"]