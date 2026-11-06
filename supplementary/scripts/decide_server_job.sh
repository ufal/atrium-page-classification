#!/bin/bash
#SBATCH -J pc-decide  # name of job
#SBATCH -p gpu-ms,gpu-troja        # name of partition or queue
#SBATCH --cpus-per-task=16                 # number of cores/threads per task
#SBATCH --gres=gpu:1                       # Clef in bf16 needs ~54 GB of GPU memory; Clef-Flash ~18 GB; Strands none
#SBATCH --mem=80G
#SBATCH --time 0-12:00:00
#SBATCH -o pc-decide.%j.%N.out
#SBATCH -e pc-decide.%j.%N.err
#
# A decision-model server on a cluster node for the #58 experiments (ufal/atrium-project#83), and the
# experiment run against it in the same job. TEMPLATE: written 2026-11-06, not yet run.
#
# Why on the cluster as well as Pavel's internal endpoint: that endpoint serves only Clef (an 8-bit
# build), and data that must stay on the cluster (the AMČR PDFs, digital-convert#8) cannot go to it.
# Here Clef runs in bf16 (the model card's setting), Clef-Flash is available, and Strands Decider 2B
# runs on the CPU.
#
#   MODEL=clef        sbatch supplementary/scripts/decide_server_job.sh   # SGLang, lmsysorg/sglang:dev-clef
#   MODEL=clef-flash  sbatch supplementary/scripts/decide_server_job.sh
#   MODEL=strands     sbatch --gres=gpu:0 supplementary/scripts/decide_server_job.sh   # strands-decider serve
#
#   bash supplementary/scripts/decide_server_job.sh prefetch   # login node, online: weights and the image
#
# The server listens on 127.0.0.1 only (strands-decider serve has no authentication), so the client
# runs in this job: DECIDE_ARGS are passed to decide_pages.py; without them the job only probes the
# server (decision_client.py --probe). Paths and choices by environment variable: ATRIUM, REPO, MODEL,
# PORT, SIF, STRANDS_VENV, DECIDE_ARGS, PROBE_IMAGES.

set -u
hostname
date

ATRIUM=${ATRIUM:-/lnet/work/projects/atrium}
REPO=${REPO:-$ATRIUM/GITHUB/atrium-page-classification}
MODEL=${MODEL:-clef}
PORT=${PORT:-30000}
SIF=${SIF:-$ATRIUM/containers/sglang_dev-clef.sif}
STRANDS_VENV=${STRANDS_VENV:-$ATRIUM/venvs/strands-decider}

case "$MODEL" in
    clef)       HF_ID=Cloudflare/clef ;;
    clef-flash) HF_ID=Cloudflare/clef-flash ;;
    strands)    HF_ID=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 ;;
    *) echo "MODEL must be clef, clef-flash or strands, not '$MODEL'" >&2; exit 2 ;;
esac

export HF_HOME=$ATRIUM/cache/
export PYTHONUNBUFFERED=1

# ── prefetch: on a login node with network, before submitting ─────────────────────────────────────
if [[ "${1:-}" == "prefetch" ]]; then
    python3 -c "from huggingface_hub import snapshot_download as s; print(s('$HF_ID'))" || exit 1
    if [[ "$MODEL" != strands && ! -f "$SIF" ]]; then
        mkdir -p "$(dirname "$SIF")"
        apptainer pull "$SIF" docker://lmsysorg/sglang:dev-clef || exit 1
    fi
    exit 0
fi

export HF_HUB_OFFLINE=1
cd "$REPO" || exit 1
# Two public pages for the image check: a photograph and a printed text page.
PROBE_IMAGES=${PROBE_IMAGES:-"small_data_samples/PHOTO/caa_conference-01.png $(find small_data_samples/TEXT_P -name '*.png' | sort | head -1)"}
STAMP=$(date +%Y%m%d-%H%M)
OUT=result/decide/${STAMP}_${MODEL}_server
mkdir -p "$OUT"

# ── provenance: what served the answers ───────────────────────────────────────────────────────────
{
    echo "# Decision server, job ${SLURM_JOB_ID:-local} on $(hostname), $(date -u +%FT%TZ)"
    echo "model: $HF_ID"
    echo "revision: $(python3 -c "from huggingface_hub import snapshot_download as s; import os; print(os.path.basename(s('$HF_ID', local_files_only=True)))" 2>/dev/null)"
    echo "repo commit: $(git rev-parse HEAD)"
    if [[ "$MODEL" == strands ]]; then
        echo "server: strands-decider $("$STRANDS_VENV/bin/pip" show strands-decider 2>/dev/null | sed -n 's/^Version: //p')"
    else
        echo "server: SGLang, $SIF ($(sha256sum "$SIF" | cut -c1-16))"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    fi
} > "$OUT/PROVENANCE.md"
cat "$OUT/PROVENANCE.md"

# ── start the server ──────────────────────────────────────────────────────────────────────────────
if [[ "$MODEL" == strands ]]; then
    "$STRANDS_VENV/bin/strands-decider" serve --model "$HF_ID" --port "$PORT" > "$OUT/server.log" 2>&1 &
else
    apptainer exec --nv "$SIF" sglang serve --model-path "$HF_ID" --host 127.0.0.1 --port "$PORT" \
        > "$OUT/server.log" 2>&1 &
fi
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

# ── wait until it answers the card's own request (cold load is measured here) ─────────────────────
T0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/systemone" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$HF_ID\",\"state\":\"ping\",\"questions\":{\"q\":{\"type\":\"noul\"}}}" > /dev/null; do
    kill -0 $SERVER_PID 2>/dev/null || { echo "the server exited; see $OUT/server.log" >&2; exit 1; }
    sleep 10
done
echo "cold load: $(( $(date +%s) - T0 )) s" | tee -a "$OUT/PROVENANCE.md"

# shellcheck source=/dev/null
source "$REPO/venv/bin/activate"
export DECISION_MODEL_URL=http://127.0.0.1:$PORT
export DECISION_MODEL_NAME=$HF_ID

probe_args=()
for image in $PROBE_IMAGES; do  # a space-separated list of paths without spaces
    probe_args+=(--image "$image")
done
python3 decision_client.py --probe --no-proxy "${probe_args[@]}" --out "$OUT/probe.json"

if [[ -n "${DECIDE_ARGS:-}" ]]; then
    # shellcheck disable=SC2086 # DECIDE_ARGS is a list of arguments
    python3 decide_pages.py --no-proxy --served-on cluster-job \
        --model-revision "$(sed -n 's/^revision: //p' "$OUT/PROVENANCE.md")" $DECIDE_ARGS
fi
date
