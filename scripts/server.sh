#!/usr/bin/env bash
# Start the ATRIUM Page Classification API server and wait until it is healthy.
#
# Prefers Docker Compose (CPU by default, GPU with --gpu), falls back to a
# local uvicorn launch inside the repository's virtual environment.
#
# Usage:
#   bash scripts/serve.sh            # Docker CPU, or local uvicorn fallback
#   bash scripts/serve.sh --gpu      # Docker with GPU (docker-compose.gpu.yml)
#   bash scripts/serve.sh --local    # skip Docker, run uvicorn directly
#
# Environment:
#   ATRIUM_PC_PORT  - port to serve on (default: 8000)
#   ATRIUM_PC_URL   - health-check target (default: http://localhost:$ATRIUM_PC_PORT)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${ATRIUM_PC_PORT:-8000}"
BASE_URL="${ATRIUM_PC_URL:-http://localhost:${PORT}}"
HEALTH_URL="${BASE_URL}/info"
MODE="auto"

for arg in "$@"; do
    case "$arg" in
        --gpu)   MODE="gpu" ;;
        --local) MODE="local" ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

# Already running? Nothing to do.
if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    echo "✅ API already healthy at ${BASE_URL}"
    exit 0
fi

cd "$REPO_ROOT"

start_docker() {
    local compose_file="$1"
    echo "🐳 Starting via docker compose (${compose_file})..."
    docker compose -f "$compose_file" up -d
}

start_local() {
    echo "🐍 Starting local uvicorn server..."
    if [ ! -d "venv" ]; then
        echo "No venv found - running setup/setup_api_service.sh first..."
        bash setup/setup_api_service.sh
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate
    nohup uvicorn service.api:app --host 0.0.0.0 --port "$PORT" > api_server.log 2>&1 &
    echo "Server PID: $! (logs: api_server.log)"
}

case "$MODE" in
    gpu)   start_docker docker-compose.gpu.yml ;;
    local) start_local ;;
    auto)
        if command -v docker > /dev/null 2>&1 && docker info > /dev/null 2>&1; then
            start_docker docker-compose.yml
        else
            start_local
        fi
        ;;
esac

# First launch downloads model weights from Hugging Face (up to ~1.2 GB per
# revision, 5 revisions for the ensemble) - allow a generous startup window.
echo "⏳ Waiting for ${HEALTH_URL} (model warmup / first-run download may take several minutes)..."
DEADLINE=$((SECONDS + 900))
until curl -sf "$HEALTH_URL" > /dev/null 2>&1; do
    if [ "$SECONDS" -ge "$DEADLINE" ]; then
        echo "❌ Server did not become healthy within 15 minutes." >&2
        echo "   Check: api_server.log (local) or 'docker compose logs' (Docker)." >&2
        exit 1
    fi
    sleep 5
done

echo "✅ API healthy at ${BASE_URL}"
