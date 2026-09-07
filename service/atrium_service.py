"""atrium_service.py — shared FastAPI meta-contract helpers for ATRIUM services.

Canonical copy lives in the hub at ``docs/templates/shared/atrium_service.py`` and is
mirrored **byte-identically** into every tool repo's ``service/`` directory (enforced by
``para-drift.reusable.yml``, the same mechanism that guards ``atrium_paradata.py``).

It implements the normative §4 meta-contract of ``docs/agent_skill_strategy.md`` so every
service reports an identical shape and agents/clients can rely on it:

* ``read_tool_version`` — version from ``para_config.txt`` ``[tool]`` (single source of truth).
* ``build_info``        — the §4.1 ``/info`` envelope (``service``/``version``/``endpoints``/``limits``).
* ``attach_health``     — the §4.1 ``GET /health`` endpoint (shallow + ``?deep=true``), and,
  when given a ``ServiceState``, the §4.6 ``GET /ready`` readiness endpoint (issue #55).
* ``resolve_max_upload_mb`` / ``add_cors`` — the §4.5 upload-limit and CORS conventions.
* ``ServiceState`` / ``attach_inflight_middleware`` / ``serve_lifecycle`` — the §4.6
  disposability contract (issue #55): readiness that flips on ``SIGTERM``, and a drain that
  waits for in-flight requests and explicitly tracked background work before the process
  exits, so a rolling restart does not kill work already in progress.

The module deliberately imports only FastAPI/Starlette and the standard library (already a
dependency of every service) so it stays inside the no-model fast lane.
"""

from __future__ import annotations

import asyncio
import configparser
import contextlib
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Paths FastAPI mounts for documentation/schema — callable, but not part of the
# domain API surface advertised by /info.
_INFRA_PATHS = {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}

#: Signals a service should treat as "start draining" (issue #55). SIGTERM is what
#: `docker stop` / a Kubernetes rolling restart send; SIGINT is Ctrl-C during local dev.
_DRAIN_SIGNALS = (signal.SIGTERM, signal.SIGINT)


def read_tool_version(start: Path | str, default: str = "0.0.0") -> str:
    """Return the ``[tool] version`` from ``para_config.txt`` (single source of truth).

    Walks ``start`` and its parents looking for ``para_config.txt`` or
    ``setup/para_config.txt`` — covering both repo layouts (page-classification and
    alto-postprocess keep it under ``setup/``; the others at the repo root). A leading
    ``v`` is stripped so ``/info`` and ``app.version`` match the CITATION/release value
    exactly. ``security.reusable.yml`` already validates that value, so the API version
    can never drift from the released version.
    """
    start = Path(start).resolve()
    for root in [start, *start.parents]:
        for candidate in (root / "para_config.txt", root / "setup" / "para_config.txt"):
            if candidate.exists():
                config = configparser.ConfigParser()
                config.read(candidate, encoding="utf-8")
                version = config.get("tool", "version", fallback=None)
                if version:
                    return version[1:] if version.lower().startswith("v") else version
    return default


def list_endpoints(app: FastAPI) -> List[str]:
    """Return the callable API paths registered on ``app``.

    Excludes the FastAPI docs/schema infrastructure and mounted sub-apps (e.g.
    ``StaticFiles`` frontends, which expose no HTTP ``methods``), so the list matches
    the domain endpoints an agent would actually call.
    """
    paths = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not path or methods is None:  # mounts / static apps have no `methods`
            continue
        if path in _INFRA_PATHS:
            continue
        paths.add(path)
    return sorted(paths)


def build_info(
    app: FastAPI,
    service: str,
    limits: Optional[Dict[str, Any]] = None,
    **capabilities: Any,
) -> Dict[str, Any]:
    """Assemble the normative §4.1 ``/info`` envelope.

    Guarantees the four required keys — ``service`` (canonical tool id == repo name),
    ``version`` (``== app.version``), ``endpoints`` (the live route set) and ``limits``
    (at least ``max_upload_mb``) — are always present. Service-specific capability
    fields (categories, supported formats, models, backends, …) are passed through as
    extra keyword arguments.
    """
    info: Dict[str, Any] = {
        "service": service,
        "version": app.version,
        "endpoints": list_endpoints(app),
        "limits": dict(limits or {}),
    }
    info.update(capabilities)
    return info


class ServiceState:
    """Readiness/draining/in-flight state for graceful shutdown (§4.6, issue #55).

    One instance per service process, created at module import time and threaded through
    ``attach_health``, ``attach_inflight_middleware`` and ``serve_lifecycle``.

    * ``warm`` — set ``True`` once startup/model-load work is complete. ``GET /ready`` is
      503 until then; this is the ``startupProbe`` target for a slow-warming service.
    * ``draining`` — set ``True`` the instant SIGTERM/SIGINT is received (by
      ``serve_lifecycle``). ``GET /ready`` flips to 503 immediately, so an orchestrator's
      readiness probe removes this pod from Service endpoints before new requests arrive.
      ``GET /health`` (liveness) deliberately stays 200 throughout draining — a liveness
      probe that fails during a graceful shutdown gets the pod SIGKILLed before the drain
      finishes, which is the opposite of what this state exists to prevent.
    * ``in_flight`` — count of requests currently being served, maintained by
      ``attach_inflight_middleware``.

    A request that hands work to something that outlives the request itself — a job queue,
    ``asyncio.create_task``, a ``starlette.background.BackgroundTask`` — is invisible to
    ``in_flight``: the counter reaches zero the moment such a request *returns*, even
    though the work it started is still running. That is what "a rolling restart kills
    in-flight work" (issue #55) actually meant for a service shaped like nlp-enrich's job
    API. Use :meth:`track` for that work so shutdown waits for it too.
    """

    def __init__(self) -> None:
        self.warm = False
        self.draining = False
        self.in_flight = 0
        self._tracked: Set[asyncio.Task] = set()

    def track(self, coro: Awaitable[Any]) -> asyncio.Task:
        """Schedule ``coro`` as a task shutdown will wait for, instead of a bare
        ``asyncio.create_task(...)``.

        Use this for any job that survives past the request that started it. The task
        reference is retained here (unlike a discarded ``asyncio.create_task(...)``
        result, which is eligible for garbage collection even with no shutdown involved)
        and dropped automatically once it finishes.
        """
        task = asyncio.ensure_future(coro)
        self._tracked.add(task)
        task.add_done_callback(self._tracked.discard)
        return task

    async def wait_drained(self, timeout: float) -> bool:
        """Wait for ``in_flight`` requests and every tracked task to finish.

        Bounded by ``timeout`` seconds (wall clock via ``time.monotonic``, not tied to any
        event loop). Returns ``True`` if fully drained, ``False`` if the timeout elapsed
        with work still outstanding — the caller should log that case; it means the grace
        period given to the container was too short, not that the wait itself failed.
        A service with nothing to track (no job queue, no detached tasks) returns
        immediately and this is a no-op.
        """
        deadline = time.monotonic() + timeout
        while (self.in_flight > 0 or self._tracked) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        return self.in_flight == 0 and not self._tracked


def attach_inflight_middleware(app: FastAPI, state: ServiceState) -> None:
    """Count requests currently being served on ``state.in_flight`` (issue #55).

    Registration order relative to other middleware does not matter — this only counts,
    it never rejects or redirects a request.
    """

    @app.middleware("http")
    async def _count_inflight(request, call_next):  # noqa: ANN001, ANN202
        state.in_flight += 1
        try:
            return await call_next(request)
        finally:
            state.in_flight -= 1


def attach_health(
    app: FastAPI,
    deep_check: Optional[Callable[[], Optional[str]]] = None,
    state: Optional[ServiceState] = None,
) -> None:
    """Register the normative §4.1 ``GET /health`` endpoint on ``app``, and — only when
    ``state`` is given — the §4.6 ``GET /ready`` readiness endpoint (issue #55).

    * Shallow (``GET /health``): cheap liveness → ``{"status": "ok"}`` HTTP 200,
      unconditionally, even while draining. This is unchanged from before ``state``
      existed and stays byte-identical to keep the five existing per-repo
      ``test_health_shallow_ok`` assertions (and ``skill-validate.reusable.yml``'s live
      probe) green without modification. Liveness is deliberately not where "stop
      routing new work here" belongs — that is ``/ready``'s job; a liveness probe
      failing during a graceful shutdown causes a SIGKILL before the drain completes.
    * Deep (``GET /health?deep=true``): runs ``deep_check`` — a callable returning
      ``None`` when healthy or a short detail string when degraded — and answers
      ``{"status": "degraded", "detail": …}`` HTTP 503 on failure, same as before. When
      ``state`` is given, deep additionally reports ``{"status": "degraded", "detail":
      "shutting down"}`` while draining (taking priority over ``deep_check``, since a
      draining process is degraded regardless of what its dependencies report), and
      always includes ``in_flight``/``draining`` in the body for operators.
    * ``GET /ready`` (only registered when ``state`` is given): 503 (``"starting"``)
      until ``state.warm``, 200 (``"ready"``) while serving, 503 (``"draining"``) the
      instant a shutdown signal is received. This is the endpoint a Kubernetes
      ``readinessProbe``/``startupProbe`` should target — it is what removes a pod from
      Service endpoints before a rolling restart's ``SIGTERM`` lands, and the
      shallow-liveness endpoint above cannot do this without breaking existing callers.

    ``deep_check`` must never raise; if it does, the failure is reported as degraded
    rather than surfacing a 500.
    """

    @app.get("/health")
    def health(deep: bool = False) -> JSONResponse:
        if not deep:
            return JSONResponse({"status": "ok"}, status_code=200)

        detail: Optional[str] = None
        if state is not None and state.draining:
            detail = "shutting down"
        elif deep_check is not None:
            try:
                detail = deep_check()
            except Exception as exc:  # a probe must never turn a health check into a 500
                detail = f"deep health check raised: {exc}"

        body: Dict[str, Any] = {"status": "degraded" if detail else "ok"}
        if detail:
            body["detail"] = detail
        if state is not None:
            body["in_flight"] = state.in_flight
            body["draining"] = state.draining
        return JSONResponse(body, status_code=503 if detail else 200)

    if state is not None:

        @app.get("/ready")
        def ready() -> JSONResponse:
            if state.draining:
                return JSONResponse({"status": "draining"}, status_code=503)
            if not state.warm:
                return JSONResponse({"status": "starting"}, status_code=503)
            return JSONResponse({"status": "ready"}, status_code=200)


@contextlib.asynccontextmanager
async def serve_lifecycle(state: ServiceState, drain_timeout: float = 25.0):
    """Async context manager an existing FastAPI ``lifespan`` wraps its body in (issue #55).

    Every ATRIUM service already has a ``lifespan`` that does warmup; this composes with
    it rather than replacing it::

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            warm_up_models()          # existing startup work, unchanged
            state.warm = True
            async with serve_lifecycle(state):
                yield
            teardown()                 # existing shutdown work, unchanged — runs AFTER
                                        # the drain below completes

    **On entry**, installs a handler for ``SIGTERM``/``SIGINT`` that sets
    ``state.draining = True`` and then calls whatever handler was previously registered
    for that signal. uvicorn installs its own handler (``Server.handle_exit``, via
    ``Server.capture_signals()``) in ``Server.serve()`` *before* it runs app startup —
    and this call happens during app startup — so the "previous" handler this captures
    is always uvicorn's own, and calling it is what keeps uvicorn's graceful shutdown
    (draining in-flight HTTP connections, then resuming this generator past its
    ``yield``) working. Installing a handler that replaces rather than chains to the
    previous one breaks shutdown outright — this was verified empirically against a
    real uvicorn 0.52 subprocess sent a real ``SIGTERM`` (issue #55): the process still
    exits with the signal's own code (128 + 15 = 143, since uvicorn's
    ``capture_signals()`` re-raises the captured signal after a graceful exit, so that a
    supervisor sees the same exit status a process that did not catch the signal at all
    would show), not 0 — a container that handles ``SIGTERM`` gracefully is still
    expected to report having been terminated by it, not to fake a code-0 exit.

    **On exit** (once the wrapped ``yield`` resumes — i.e. once uvicorn's own drain of
    in-flight HTTP connections has already finished or hit its own
    ``--timeout-graceful-shutdown``): additionally waits up to ``drain_timeout`` seconds
    for ``state.in_flight`` and every task registered via :meth:`ServiceState.track` to
    reach zero, so work that outlives its originating request (a job queue, a
    ``BackgroundTask``-style cleanup) is not orphaned when the process exits. Callers
    should keep ``drain_timeout`` comfortably below the container's
    ``terminationGracePeriodSeconds`` (Kubernetes) so this wait itself does not run past
    the point a ``SIGKILL`` arrives anyway. A service with no tracked work returns
    immediately here — this is a no-op, not an added delay.

    Only installs signal handlers when called from the main thread (mirrors uvicorn's
    own guard in ``capture_signals()``), so this is also safe to call from a worker
    thread or under ``TestClient`` — it becomes a no-op wrapper in that case.
    """
    previous_handlers: Dict[int, Any] = {}
    is_main_thread = threading.current_thread() is threading.main_thread()

    if is_main_thread:

        def _make_handler(sig: int) -> Callable[[int, Any], None]:
            def _handler(signum: int, frame: Any) -> None:
                state.draining = True
                previous = previous_handlers.get(sig)
                if callable(previous):
                    previous(signum, frame)

            return _handler

        for sig in _DRAIN_SIGNALS:
            previous_handlers[sig] = signal.signal(sig, _make_handler(sig))

    try:
        yield
    finally:
        drained = await state.wait_drained(drain_timeout)
        if not drained:
            # Nothing more this module can do — no logger is wired here so a caller's
            # own logging stays the single place shutdown is reported. The grace period
            # was too short for the work outstanding; a SIGKILL follows shortly after.
            pass
        if is_main_thread:
            for sig, previous in previous_handlers.items():
                # Best-effort restore. Under uvicorn this is moot: capture_signals()'s own
                # `finally` restores ITS pre-startup snapshot right after this coroutine's
                # caller (Server.shutdown()) returns, regardless of what is installed here.
                # It matters only for a non-uvicorn caller (e.g. a direct test of this
                # context manager) that should not leak a handler past this block.
                try:
                    signal.signal(sig, previous)
                except (ValueError, TypeError):
                    # ValueError: not the main thread after all (race, defensive only).
                    # TypeError: previous was SIG_DFL/SIG_IGN's sentinel in a form
                    # signal.signal rejects on this platform — leave it installed rather
                    # than raise out of a shutdown path.
                    pass


def resolve_max_upload_mb(default_mb: float) -> float:
    """Resolve the canonical upload limit in **megabytes** (§4.5).

    Prefers ``MAX_UPLOAD_MB``; falls back to the deprecated ``MAX_UPLOAD_BYTES`` (kept
    working for one release, e.g. translator's existing env) before the built-in default.
    """
    raw_mb = os.getenv("MAX_UPLOAD_MB")
    if raw_mb is not None:
        try:
            return float(raw_mb)
        except ValueError:
            pass
    legacy_bytes = os.getenv("MAX_UPLOAD_BYTES")
    if legacy_bytes is not None:
        try:
            return float(legacy_bytes) / (1024 * 1024)
        except ValueError:
            pass
    return float(default_mb)


def allowed_origins(default: str = "*") -> List[str]:
    """Parse ``ALLOWED_ORIGINS`` (CSV) into a list; default single wildcard (§4.5)."""
    return [o.strip() for o in os.getenv("ALLOWED_ORIGINS", default).split(",") if o.strip()]


def add_cors(
    app: FastAPI,
    methods: Optional[Iterable[str]] = None,
    default_origins: str = "*",
) -> None:
    """Attach the standard CORS middleware (§4.5).

    Origins come from ``ALLOWED_ORIGINS`` (CSV, default ``*``). Credentials are enabled
    only when the origin list is not the bare ``*`` wildcard — browsers reject the
    wildcard+credentials combination.
    """
    origins = allowed_origins(default_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=list(methods) if methods else ["*"],
        allow_headers=["*"],
    )
