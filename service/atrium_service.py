"""atrium_service.py — shared FastAPI meta-contract helpers for ATRIUM services.

Canonical copy lives in the hub at ``docs/templates/shared/atrium_service.py`` and is
mirrored **byte-identically** into every tool repo's ``service/`` directory (enforced by
``para-drift.reusable.yml``, the same mechanism that guards ``atrium_paradata.py``).

It implements the normative §4 meta-contract of ``docs/agent_skill_strategy.md`` so every
service reports an identical shape and agents/clients can rely on it:

* ``read_tool_version`` — version from ``para_config.txt`` ``[tool]`` (single source of truth).
* ``build_info``        — the §4.1 ``/info`` envelope (``service``/``version``/``endpoints``/``limits``).
* ``attach_health``     — the §4.1 ``GET /health`` endpoint (shallow + ``?deep=true``).
* ``resolve_max_upload_mb`` / ``add_cors`` — the §4.5 upload-limit and CORS conventions.

The module deliberately imports only FastAPI/Starlette (already a dependency of every
service) so it stays inside the no-model fast lane.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Paths FastAPI mounts for documentation/schema — callable, but not part of the
# domain API surface advertised by /info.
_INFRA_PATHS = {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}


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


def attach_health(
    app: FastAPI,
    deep_check: Optional[Callable[[], Optional[str]]] = None,
) -> None:
    """Register the normative §4.1 ``GET /health`` endpoint on ``app``.

    * Shallow (``GET /health``): cheap liveness → ``{"status": "ok"}`` HTTP 200.
    * Deep (``GET /health?deep=true``): additionally runs ``deep_check`` — a callable
      returning ``None`` when healthy or a short detail string when degraded — and
      answers ``{"status": "degraded", "detail": …}`` HTTP 503 on failure.

    ``deep_check`` must never raise; if it does, the failure is reported as degraded
    rather than surfacing a 500.
    """

    @app.get("/health")
    def health(deep: bool = False) -> JSONResponse:
        if deep and deep_check is not None:
            try:
                detail = deep_check()
            except Exception as exc:  # a probe must never turn a health check into a 500
                detail = f"deep health check raised: {exc}"
            if detail:
                return JSONResponse({"status": "degraded", "detail": detail}, status_code=503)
        return JSONResponse({"status": "ok"}, status_code=200)


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
