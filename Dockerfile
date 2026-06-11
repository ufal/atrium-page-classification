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

# GPU (CUDA 12.1) torch by default — CPU-only is not supported for inference.
# To build a CPU image explicitly pass --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
RUN pip install --index-url ${TORCH_INDEX_URL} torch torchvision

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

ENTRYPOINT ["sh", "-c", "python3 /app/gpu_check.py && exec python3 run.py \"$@\"", "--"]
CMD ["-d", "/data/input", "--hf", "-rev", "v4.3"]