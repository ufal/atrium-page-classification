"""healthcheck.py — the Docker ``HEALTHCHECK`` probe body for ATRIUM API services (issue #55).

Canonical copy lives in the hub at ``docs/templates/shared/healthcheck.py`` and is mirrored
**byte-identically** into every tool repo's ``service/`` directory (enforced by
``para-drift.reusable.yml``, the same mechanism guarding ``atrium_service.py``).

WHY THIS EXISTS RATHER THAN A ``curl``/``wget`` ONE-LINER: none of the five ATRIUM runtime
images install ``curl``, and only alto-postprocess installs ``wget`` (for its build-time
weight fetch, not present for that reason in the other four). Adding either means a new
``apt-get`` layer — and a new package to keep patched — in five Dockerfiles, when every one
of them already has a ``python`` interpreter as PID 1's own runtime. This script uses only
``urllib.request`` from the standard library, so it works unmodified in every image.

Usage, inside a Dockerfile's ``api`` stage::

    STOPSIGNAL SIGTERM
    HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \\
        CMD ["python", "/app/service/healthcheck.py"]

Targets the shallow ``GET /health`` (liveness), never ``?deep=true`` — a deep dependency
check running *inside* the healthcheck interval risks a slow-but-recovering upstream
turning into a container restart, which is exactly the failure mode a liveness probe
should not introduce. ``--start-period`` is deliberately generous (this suite's services
warm models at startup, sometimes over a slow HF cache-miss) so a first-run download is
not read as three consecutive failures.

Configurable via the same environment variables the services themselves already read
(``HOST``/``PORT``, where set) plus ``HEALTHCHECK_PATH`` for pointing at something other
than ``/health`` in a one-off diagnostic run — normal use needs neither.

Exit code 0 on HTTP 200; exit code 1 on anything else (non-200 status, connection refused,
timeout, malformed response) — the two states Docker's ``HEALTHCHECK`` distinguishes.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

DEFAULT_PORT = "8000"
DEFAULT_PATH = "/health"
DEFAULT_TIMEOUT_S = 5.0


def main() -> int:
    host = "127.0.0.1"  # always probe the local process, never the published interface
    port = os.getenv("PORT", DEFAULT_PORT)
    path = os.getenv("HEALTHCHECK_PATH", DEFAULT_PATH)
    url = f"http://{host}:{port}{path}"

    try:
        with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT_S) as response:
            status = response.status
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        print(f"unhealthy: {url} unreachable: {exc}", file=sys.stderr)
        return 1

    if status != 200:
        print(f"unhealthy: {url} returned HTTP {status}", file=sys.stderr)
        return 1

    print(f"healthy: {url} returned HTTP 200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
