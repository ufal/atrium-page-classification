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

# ── Distro security patches, applied at build time ───────────────────────────
# `python:3.11-slim` is a floating TAG, and nothing in this ecosystem bumps it:
# no repo declares a `docker` dependabot ecosystem (docker_gha_roadmap.md, H6),
# so the base layer is whatever Docker Hub last rebuilt. On 2026-09-13 that layer
# carried perl-base 5.40.1-6 with three FIXABLE CRITICAL CVEs — CVE-2026-13221,
# CVE-2026-42496 and CVE-2026-8376, all fixed in 5.40.1-6+deb13u1. The release
# gate in atrium-project's docker-tool.reusable.yml ("Fail the release on fixable
# CRITICAL vulnerabilities") blocks on exactly that class, and because the
# promotion step is `if: success()`, a blocked release publishes by DIGEST ONLY —
# the `:<version>` and `:latest` tags are never applied.
#
# It has already cost two releases: translator v1.0.0-beta (2026-09-13, both
# targets) and nlp-enrich v0.20.2 (2026-09-15, run 34970419474, all three
# targets). THIS repo had not been tagged since, which is the only reason it had
# not happened here too — the gate is `if: startsWith(github.ref, 'refs/tags/')`,
# so day-to-day `test` pushes never surface it. (atrium-project#53)
#
# `upgrade` rather than `install --only-upgrade perl-base`, deliberately. The gate
# blocks on *fixable* CRITICALs — precisely those the distro already ships a patch
# for — so the fix that matches the gate's own definition is "apply the distro's
# available patches", not a package name that has to be edited by hand the next
# time a different one is announced.
#
# CACHE INTERACTION, which is what makes this hold rather than run once: the build
# uses `cache-from: type=gha`, so an apt layer high in the file would be served
# from cache forever and silently stop patching. It sits HERE, immediately after
# the ENV block that embeds ATRIUM_RUNNER_REF, because CI passes that as
# `github.ref_name` — a value unique to each release tag. The ENV layer therefore
# changes on every release, busting this layer with it, so every released image is
# scanned against a freshly patched base while day-to-day `test` pushes still hit
# the cache. Do not move this above the ENV block.
#
# One apt layer, not two: the upgrade and the install share a single `apt-get
# update`, so the package lists are fetched once and removed once.
# Guarded by tests/test_dockerfile_security_layer.py (atrium-project#53).
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        build-essential g++ libgl1 libglib2.0-0 ca-certificates \
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
# setup/requirements.txt only; service/requirements.txt deliberately carries no
# model stack, and the image installs both. transformers 5.x constructs models on
# the `meta` device, which crashes the timm builders for the RegNetY (v4.3) and
# EfficientNetV2 (v1.3) checkpoints used by --best; the <5 pin is the meta-device
# fix, so keep it in setup/requirements.txt.
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --index-url ${TORCH_INDEX_URL} torch==2.7.1 torchvision==0.22.1

COPY setup/requirements.txt setup/requirements-test.txt ./
COPY service/requirements.txt ./service-requirements.txt
RUN pip install -r requirements.txt -r service-requirements.txt -r requirements-test.txt

COPY . .

# REVIEW FIX (Blocker A/H): the GPU report is INFORMATIONAL only — it reports
# whether a CUDA device is visible and never aborts.  CPU inference is a
# supported (if slow) path, so the container must start regardless.  The GPU
# overlay still requests the device; if it is missing the run simply proceeds
# on CPU with a clear notice.
#
# (issue #55) This is now an ENTRYPOINT that ends in os.execv rather than a
# standalone gpu_info.py invoked from `sh -c`. The old form was
#     ENTRYPOINT ["sh", "-c", "python3 /app/gpu_info.py; exec python3 run.py \"$@\"", "--"]
# which made PID 1 /bin/sh for the duration of the GPU report — and /bin/sh does not
# forward signals, so a SIGTERM arriving in that window was silently dropped. Here
# python3 is PID 1 from the first instruction, and os.execv REPLACES it in place, so
# run.py inherits PID 1 and receives signals directly. No shell is involved at all.
RUN printf '%s\n' \
    'import os' \
    'import sys' \
    'try:' \
    '    import torch' \
    '    if torch.cuda.is_available():' \
    '        print(f"[gpu] CUDA device detected: {torch.cuda.get_device_name(0)}", flush=True)' \
    '    else:' \
    '        print("[gpu] No CUDA device visible — running on CPU (slower).", flush=True)' \
    '        print("[gpu] For GPU: build with TORCH_INDEX_URL=...cu126 and use docker-compose.gpu.yml.", flush=True)' \
    'except Exception as exc:' \
    '    print(f"[gpu] GPU probe unavailable ({exc}) — continuing.", flush=True)' \
    'os.execv(sys.executable, [sys.executable, "/app/run.py", *sys.argv[1:]])' \
    > /app/entrypoint.py

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
# Pure exec form — no shell, so python3 is PID 1 and signals reach it directly
# (issue #55). CMD args are forwarded to run.py by entrypoint.py's os.execv, so the
# documented override still works: docker compose run --rm classify --dir --inner --best
ENTRYPOINT ["python3", "/app/entrypoint.py"]
CMD ["-d", "/data/input", "--hf", "-rev", "v4.3"]


# ---------------------------------------------------------------------------
# API surface — published as :<version>-api (issue #55)
#
# Before this stage existed, page-classification's FastAPI service was reachable
# only through a docker-compose `entrypoint:` override on the BATCH image, so no
# runnable API image was ever published and there was nothing for ARÚP/ARÚB to
# deploy on Kubernetes. service/requirements.txt is already installed in `base`
# (see the pip install above), so this stage only has to declare how to serve.
# ---------------------------------------------------------------------------
FROM base AS api

# EXPOSE tracks the DEFAULT port: it is image metadata and cannot read $PORT at
# runtime. Set PORT to move the listener, and publish with `-p <port>:<port>` to
# match. (issue #58)
EXPOSE 8000

# STOPSIGNAL is the default (SIGTERM) — declared explicitly so a future edit cannot
# change it silently; service/api.py's lifespan chains to uvicorn's own handler for it
# via serve_lifecycle (service/atrium_service.py).
STOPSIGNAL SIGTERM

# PORT and HOST are read by service/api.py's __main__ block; PORT is also the port
# service/healthcheck.py probes, which is why setting it used to make the container
# permanently unhealthy — the probe moved and the listener did not. Declared here so
# `docker inspect` is self-documenting and so the probe still has a value if the code
# default ever drifts. (issue #58)
#
# GRACEFUL_SHUTDOWN_S carries the `--timeout-graceful-shutdown 20` that used to sit on
# the ENTRYPOINT line. It bounds uvicorn's wait for in-flight requests.
# Note a /predict_document call classifies up to MAX_PDF_PAGES pages sequentially, so a
# large PDF can legitimately outlive this budget — raise it together with the
# deployment's grace period for that workload (docs/k8s_deployment.md, "Known limits").
ENV PORT=8000 GRACEFUL_SHUTDOWN_S=20

# `python -m service.api`, NOT `python service/api.py`: a script launch puts
# sys.path[0] at /app/service with no package context, so the relative imports fall back to
# their bare form and `model_registry` resolves only via service/inference.py's own
# sys.path append — a chain this stage must not depend on. `-m` keeps sys.path[0] at /app — byte for byte the environment the old
# `uvicorn service.api:app` entrypoint ran in, so every repo-root import still
# resolves. (issue #58)
ENTRYPOINT ["python", "-m", "service.api"]
CMD []
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "/app/service/healthcheck.py"]
