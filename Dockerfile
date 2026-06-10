# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

# ── Provenance: build-args → ENV → atrium_paradata.py (no app code changes) ──
ARG ATRIUM_RUNNER_IMAGE=""
ARG ATRIUM_RUNNER_REPO="https://github.com/ufal/atrium-page-classification"
ARG ATRIUM_RUNNER_REF=""
ENV ATRIUM_RUNNER_IMAGE=${ATRIUM_RUNNER_IMAGE} \
    ATRIUM_RUNNER_REPO=${ATRIUM_RUNNER_REPO} \
    ATRIUM_RUNNER_REF=${ATRIUM_RUNNER_REF} \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

# libgl1/libglib2.0-0: opencv (via ultralytics).  poppler-utils: pdf2image (service API).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ libgl1 libglib2.0-0 poppler-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch FIRST so the default image doesn't pull multi-GB CUDA wheels.
# Override for a GPU image:  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --index-url ${TORCH_INDEX_URL} torch torchvision

# Dependency layers cached independently of source.  root & service requirements
# share a basename, so the service one is renamed on copy to avoid clobbering.
COPY requirements.txt requirements-test.txt ./
COPY service/requirements.txt ./service-requirements.txt
RUN pip install -r requirements.txt -r service-requirements.txt -r requirements-test.txt

COPY . .

RUN useradd --create-home --uid 10001 atrium \
    && mkdir -p /cache/huggingface /data /app/model /app/result \
    && chown -R atrium:atrium /app /cache /data
USER atrium

# Batch CLI by default: classify a mounted folder, auto-download latest ViT from HF.
ENTRYPOINT ["python3", "run.py"]
CMD ["-d", "/data/input", "--hf", "-rev", "v5.3"]
