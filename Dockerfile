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

# REVIEW FIX (Blocker A/H): CPU is now the DEFAULT build.  The README, the
# service README, and the code (classifier.py / service/inference.py both fall
# back to torch.device('cpu')) all support CPU inference, so the image must not
# hard-block it.  For a CUDA image, build with the cu126 wheel index AND run
# with the GPU overlay (docker-compose.gpu.yml):
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 .
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm classify
#
# torch/torchvision are pinned to match the local environment (2.7.1 / 0.22.1)
# and installed FIRST so the later `pip install -r requirements.txt` sees them
# already satisfied and does not pull a different wheel from PyPI.
#
# NOTE: `transformers<5` is intentionally NOT pinned here — it is pinned in
# requirements.txt (and service/requirements.txt). transformers 5.x constructs
# models on the `meta` device, which crashes the timm builders for the RegNetY
# (v4.3) and EfficientNetV2 (v1.3) checkpoints used by --best; the <5 pin is the
# meta-device fix and must stay consistent across every requirements file.
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --index-url ${TORCH_INDEX_URL} torch==2.7.1 torchvision==0.22.1

COPY requirements.txt requirements-test.txt ./
COPY service/requirements.txt ./service-requirements.txt
RUN pip install -r requirements.txt -r service-requirements.txt -r requirements-test.txt

COPY . .

# REVIEW FIX (Blocker A/H): gpu_info.py is now INFORMATIONAL only — it reports
# whether a CUDA device is visible and never aborts.  CPU inference is a
# supported (if slow) path, so the container must start regardless.  The GPU
# overlay still requests the device; if it is missing the run simply proceeds
# on CPU with a clear notice.
RUN printf '%s\n' \
    'import torch' \
    'if torch.cuda.is_available():' \
    '    print(f"[gpu] CUDA device detected: {torch.cuda.get_device_name(0)}")' \
    'else:' \
    '    print("[gpu] No CUDA device visible — running on CPU (slower).", flush=True)' \
    '    print("[gpu] For GPU: build with TORCH_INDEX_URL=...cu126 and use docker-compose.gpu.yml.", flush=True)' \
    > /app/gpu_info.py

RUN useradd --create-home --uid 10001 atrium \
    && mkdir -p /cache/huggingface /data /app/model /app/result \
    && chown -R atrium:atrium /app /cache /data
USER atrium

# Default: single-model directory inference (v4.3).  Args pass straight through
# to run.py, so the memory-aware ensemble engine is available too, e.g.:
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
#       run --rm classify --dir --inner --best --parallel
# The GPU profile registry (model/gpu_profile.json) is written into the
# page-model volume and is hardware/torch/batch-keyed, so it is reused across
# runs on the same GPU and auto-invalidated (re-profiled) on a different one.
ENTRYPOINT ["sh", "-c", "python3 /app/gpu_info.py; exec python3 run.py \"$@\"", "--"]
CMD ["-d", "/data/input", "--hf", "-rev", "v4.3"]