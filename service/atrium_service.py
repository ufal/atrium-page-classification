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
* ``attach_error_handlers`` / ``error_body`` / ``AtriumHTTPError`` / ``busy`` — the §4.4
  harmonised error body ``{status, reason, detail}`` with its registered ``reason`` codes
  (atrium-project#32 item 2, landed with #53). Every error a service returns has this
  shape; ``reason`` is ``null`` until a code is registered for its cause.
* ``build_info`` given a ``LimitSet`` and ``read_upload_bounded`` / ``check_body_size`` —
  the §4.5 limits contract (atrium-project#53, factor III): every limit is a setting,
  reported in ``/info`` with the variable that sets it, and an input over one is refused
  with ``reason: "limit_exceeded"``. The limits themselves are declared with the
  framework-free ``atrium_limits.py`` at the repo root, which library code and the CLIs
  import directly; that module is imported lazily here, inside the functions that need
  it, because a service may import this module before its repo root is on ``sys.path``.
* ``ServiceState`` / ``attach_inflight_middleware`` / ``serve_lifecycle`` — the §4.6
  disposability contract (issue #55): readiness that flips on ``SIGTERM``, and a drain that
  waits for in-flight requests and explicitly tracked background work before the process
  exits, so a rolling restart does not kill work already in progress.
* The typed contract (§4.8; atrium-project#32 round 2, items 1 and 3): the pydantic models
  every service's responses are declared with (``ErrorBody``, ``InfoBase``, ``LimitNote``,
  ``HealthBody``, ``ReadyBody``, ``CreateAction``, ``AtriumDocument``); ``error_responses``
  and ``operation_id``, which every ``FastAPI(...)`` passes so each route documents the
  error body and keeps a stable operationId; ``attach_openapi_contract``, which finishes the
  generated spec (the reason-code registry, the record schema, the service id); and
  ``openapi_digest``, the sha256 ``/info`` reports so a deployed image can be checked
  against the ``openapi.json`` attached to its release. The spec is committed as
  ``service/openapi.json`` and released by the stdlib-only ``atrium_openapi.py`` at the repo
  root, which this module imports lazily, like ``atrium_limits``.
* ``parse_record_part`` — opens the document record a request carries and refuses one that
  cannot be opened with ``reason: "invalid_record"``; a record that opens but fails the
  schema is left to the tool, which reports it as ``document_json_schema_error``.

The module deliberately imports only FastAPI/Starlette, pydantic (FastAPI's own model layer,
declared and pinned in every ``service/requirements.txt``, because the committed spec is a
function of both versions) and the standard library, so it stays inside the no-model fast
lane.
"""

from __future__ import annotations

import asyncio
import configparser
import contextlib
import json
import os
import signal
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_serializer

# Paths FastAPI mounts for documentation/schema — callable, but not part of the
# domain API surface advertised by /info.
_INFRA_PATHS = {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}

#: Signals a service should treat as "start draining" (issue #55). SIGTERM is what
#: `docker stop` / a Kubernetes rolling restart send; SIGINT is Ctrl-C during local dev.
_DRAIN_SIGNALS = (signal.SIGTERM, signal.SIGINT)

#: Read size for :func:`read_upload_bounded` — bounds how far a read may overshoot a limit.
_UPLOAD_CHUNK_BYTES = 1024 * 1024


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
    limits: Any = None,
    **capabilities: Any,
) -> Dict[str, Any]:
    """Assemble the normative §4.1 ``/info`` envelope.

    Guarantees the four required keys — ``service`` (canonical tool id == repo name),
    ``version`` (``== app.version``), ``endpoints`` (the live route set) and ``limits``
    (at least ``max_upload_mb``) — are always present. Service-specific capability
    fields (categories, supported formats, models, backends, …) are passed through as
    extra keyword arguments.

    ``limits`` is either a plain ``{key: value}`` mapping (unchanged behaviour) or the
    repo's ``atrium_limits.LimitSet`` (atrium-project#53). Given a ``LimitSet``,
    ``limits`` is its flat map of effective values — the same shape as before, so
    existing readers of ``limits.max_upload_mb`` are unaffected — and a ``limits_meta``
    key is added: for each limit, the environment variable that sets it, its unit,
    default and where the current value came from (``env``/``config``/``default``/
    ``derived``), so a caller knows which setting to change.

    ``openapi_sha256`` (atrium-project#32 item 3) is :func:`openapi_digest` of the spec this
    process serves, so a client can check a running image against the ``openapi.json`` and
    ``openapi.json.sha256`` attached to the release it trusts without comparing the documents.
    """
    meta: Optional[Dict[str, Any]] = None
    if limits is None:
        values: Dict[str, Any] = {}
    elif isinstance(limits, Mapping):
        values = dict(limits)
    else:
        values = dict(limits.values())
        meta = limits.meta()
    info: Dict[str, Any] = {
        "service": service,
        "version": app.version,
        "endpoints": list_endpoints(app),
        "limits": values,
    }
    if meta is not None:
        info["limits_meta"] = meta
    info["openapi_sha256"] = openapi_digest(app)
    info.update(capabilities)
    return info


# ──────────────────────────────────────────────────────────────────────────────
# The harmonised error body (§4.4; atrium-project#32 item 2, landed with #53)
# ──────────────────────────────────────────────────────────────────────────────

#: The registered ``reason`` codes. A published code is never renamed or removed; a new
#: cause gets a new code. ``null`` means no code is registered for the cause yet.
#:
#: The registry is published in every service's spec as ``x-atrium-reason-codes`` (see
#: :func:`attach_openapi_contract`), and ``atrium_openapi.py compare`` fails a release whose
#: spec drops a code the previous release had. ``reason`` itself stays an open string in the
#: spec, not an enum, so adding a code never breaks a client generated from an older spec.
#: The first three codes below the original two were registered by atrium-project#32 round 2:
#: ``unsupported_media_type`` (the media-type refusals of all five services),
#: ``invalid_record`` (the record a request carries, the #67 R1 seed included) and
#: ``ocr_text_layer`` (the born-digital refusal the AMČR route step sends to OCR,
#: atrium-llm-enrich#10 W6; raised by ``api-digital``).
REASON_CODES: Dict[str, str] = {
    "limit_exceeded": (
        "The input is over one of the service's limits. The body's `limit` member names it "
        "(`key`, the environment variable `env` that sets it, `value`, `observed`, `unit`). "
        "HTTP 413 when the request content is too large, 422 when a parameter or a per-input "
        "processing budget is exceeded, 504 when a time limit that depends on an upstream "
        "service expired. Split the input, or raise the limit."
    ),
    "busy": (
        "Every processing slot or queue place is taken (HTTP 429). Retry after the "
        "`Retry-After` header's number of seconds."
    ),
    "unsupported_media_type": (
        "The upload is not a type this endpoint reads (HTTP 415). `detail` says what was sent; "
        "the body's `accepted` member lists the file extensions or media types the endpoint "
        "accepts, and `cause` may name a finer, unregistered reason. Convert the input or send "
        "it to the tool that reads it; do not retry it unchanged."
    ),
    "invalid_record": (
        "The document record sent with the request cannot be opened (HTTP 422): it is not UTF-8 "
        "JSON, not a JSON object, or its `schema_version` has a newer major than this tool reads. "
        "A record that opens but does not validate against the schema is accepted and reported "
        "in `document_json_schema_error` instead. Fix the record; do not retry it unchanged."
    ),
    "ocr_text_layer": (
        "The PDF's text layer is an earlier OCR run's invisible text over page images, so the "
        "document is not born-digital (HTTP 422). Route it to OCR: that text layer is not trusted."
    ),
}

#: The HTTP statuses each registered code may be sent with (§4.4). ``error_body`` and
#: :class:`AtriumHTTPError` refuse any other pairing, so a client can rely on the pair.
#: ``limit_exceeded``'s statuses are ``atrium_limits.LIMIT_STATUSES`` (the hub's
#: tests/test_atrium_service.py holds the two equal).
REASON_STATUSES: Dict[str, Tuple[int, ...]] = {
    "limit_exceeded": (413, 422, 504),
    "busy": (429,),
    "unsupported_media_type": (415,),
    "invalid_record": (422,),
    "ocr_text_layer": (422,),
}


def _check_reason(where: str, status: int, reason: Optional[str]) -> None:
    if reason is None:
        return
    if reason not in REASON_CODES:
        raise ValueError(f"{where}: {reason!r} is not a registered reason code {sorted(REASON_CODES)}")
    if int(status) not in REASON_STATUSES[reason]:
        raise ValueError(f"{where}: {reason!r} is sent with HTTP {REASON_STATUSES[reason]}, not {status}")


def error_body(status: int, detail: str, reason: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """The §4.4 error body: ``{status, reason, detail}`` plus optional members.

    ``status`` is the HTTP status as an integer. It appears only on error bodies: the
    string ``status`` of ``/health``, ``/ready`` and a job resource is a different field
    of a different response. ``detail`` is always a human-readable string (existing
    clients read it); structured data goes into extra members (``limit``, ``errors``,
    ``accepted``, ``cause``). A ``reason`` must be registered, and registered for ``status``.
    """
    _check_reason("error_body()", status, reason)
    body: Dict[str, Any] = {"status": int(status), "reason": reason, "detail": str(detail)}
    body.update(extra)
    return body


class AtriumHTTPError(HTTPException):
    """An ``HTTPException`` that carries a registered ``reason`` and extra body members."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        reason: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        **extra: Any,
    ) -> None:
        _check_reason("AtriumHTTPError", status_code, reason)
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.reason = reason
        self.extra = extra


def busy(detail: str = "Server busy; every processing slot is taken.", *, retry_after_s: int = 5) -> AtriumHTTPError:
    """The §4.4 ``busy`` refusal: HTTP 429 with ``Retry-After`` (seconds).

    Only 429: §4.4 tells clients to retry a 503 three times with backoff, which is the
    right answer to a draining replica, not to a full one. A draining replica's 503
    keeps ``reason: null``.
    """
    return AtriumHTTPError(429, detail, reason="busy", headers={"Retry-After": str(int(retry_after_s))})


def _status_phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def attach_error_handlers(app: FastAPI) -> None:
    """Give every error ``app`` returns the §4.4 body ``{status, reason, detail}``.

    * Starlette's ``HTTPException`` — FastAPI's subclass, :class:`AtriumHTTPError`, and
      the router's own 404/405 — keeps its status, headers and ``detail`` string;
      ``reason`` and extra members come from an :class:`AtriumHTTPError`, else ``null``.
      A non-string ``detail`` moves to ``errors`` and ``detail`` becomes the status
      phrase, so ``detail`` is always a string.
    * ``RequestValidationError`` → 422 with ``detail`` naming the first problem and
      FastAPI's list of problems in ``errors``.
    * ``atrium_limits.LimitExceeded`` → its own status (413/422/504) with
      ``reason: "limit_exceeded"`` and the ``limit`` member.
    * Anything else → 500 ``{"status": 500, "reason": null, "detail": "Internal server
      error."}`` as JSON instead of a text/plain page. Starlette still re-raises it
      after the response, so the server logs the traceback as before.

    ``detail`` strings are unchanged, so clients and tests that read ``["detail"]`` keep
    working. ``/health`` and ``/ready`` answer with their own bodies, not through here.
    """
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError

    import atrium_limits  # lazily: see the module docstring

    # The router's own 404/405 raise Starlette's HTTPException, the base class of FastAPI's.
    # It is taken from FastAPI's class rather than imported from `starlette`, which no
    # service declares as a dependency of its own (page-classification's
    # tests/test_service_runtime_deps.py fails a service module that imports it).
    StarletteHTTPException = next(cls for cls in HTTPException.__mro__[1:] if cls.__name__ == "HTTPException")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request, exc):  # noqa: ANN001, ANN202
        reason = getattr(exc, "reason", None)
        extra: Dict[str, Any] = dict(getattr(exc, "extra", {}) or {})
        detail = exc.detail
        if not isinstance(detail, str):
            extra.setdefault("errors", jsonable_encoder(detail))
            detail = _status_phrase(exc.status_code)
        body = error_body(exc.status_code, detail, reason, **extra)
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):  # noqa: ANN001, ANN202
        errors = jsonable_encoder(exc.errors())
        first = errors[0] if errors else {}
        where = ".".join(str(part) for part in first.get("loc", []) if part != "body")
        msg = first.get("msg", "invalid request")
        detail = f"Request validation failed: {where + ': ' if where else ''}{msg}."
        return JSONResponse(error_body(422, detail, None, errors=errors), status_code=422)

    @app.exception_handler(atrium_limits.LimitExceeded)
    async def _limit_exceeded(request, exc):  # noqa: ANN001, ANN202
        body = error_body(exc.http_status, exc.detail, "limit_exceeded", limit=jsonable_encoder(exc.to_dict()))
        return JSONResponse(body, status_code=exc.http_status)

    @app.exception_handler(Exception)
    async def _unhandled(request, exc):  # noqa: ANN001, ANN202
        return JSONResponse(error_body(500, "Internal server error."), status_code=500)


async def read_upload_bounded(upload: Any, max_mb: float, label: str = "File") -> bytes:
    """Read an ``UploadFile`` in 1 MiB chunks, refusing it once it is over ``max_mb``.

    Raises ``atrium_limits.LimitExceeded`` (413, key ``max_upload_mb``) instead of
    reading the whole part into memory first. Use it for every uploaded part a service
    keeps — the document record too, not only the main file.
    """
    import atrium_limits  # lazily: see the module docstring

    limit_bytes = int(max_mb * 1024 * 1024)
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit_bytes:
            size = getattr(upload, "size", None)
            observed = round(size / (1024 * 1024), 2) if isinstance(size, int) and size > limit_bytes else None
            raise atrium_limits.LimitExceeded(
                "max_upload_mb",
                max_mb,
                observed,
                unit="MB",
                env="MAX_UPLOAD_MB",
                detail=f"{label} too large: over {max_mb:g} MB (MAX_UPLOAD_MB).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def check_body_size(request: Any, max_mb: float, label: str = "Request body") -> None:
    """Refuse a request body over ``max_mb`` (for JSON endpoints, which have no upload).

    Uses ``Content-Length`` when the client sent one, else the body FastAPI has already
    read (``Request.body()`` is cached, so this reads nothing twice).
    """
    import atrium_limits  # lazily: see the module docstring

    limit_bytes = int(max_mb * 1024 * 1024)
    size: Optional[int] = None
    header = request.headers.get("content-length")
    if header and header.isdigit():
        size = int(header)
    if size is None:
        size = len(await request.body())
    if size > limit_bytes:
        raise atrium_limits.LimitExceeded(
            "max_upload_mb",
            max_mb,
            round(size / (1024 * 1024), 2),
            unit="MB",
            env="MAX_UPLOAD_MB",
            detail=f"{label} too large: over {max_mb:g} MB (MAX_UPLOAD_MB).",
        )


def parse_record_part(raw: Any, label: str = "document_json") -> Optional[Dict[str, Any]]:
    """Open the document record a request carries, or refuse it with ``invalid_record``.

    ``raw`` is the part's bytes (or text), an already decoded JSON value from a JSON body,
    or ``None``. An absent or empty part returns ``None``: an HTML form sends an empty file
    part when nothing was chosen, and nlp-enrich already treated one as absent.

    Refused, HTTP 422 ``reason: "invalid_record"`` (§4.4), are only a record that cannot be
    opened at all: not UTF-8 JSON, not a JSON object, or a ``schema_version`` whose major is
    newer than this module's ``atrium_document.SCHEMA_VERSION``. A record that opens but
    does not validate against the schema is returned as it is: every tool accepts such a
    baseline and reports the problem as ``document_json_schema_error``, and a #67 R1 seed
    (``doc_id`` and ``source`` only) must stay valid input.
    """
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise AtriumHTTPError(422, f"The {label} part is not UTF-8 text.", reason="invalid_record") from exc
    if isinstance(raw, str):
        raw = raw[1:] if raw.startswith("\ufeff") else raw
        if not raw.strip():
            return None
        try:
            record = json.loads(raw)
        except ValueError as exc:
            raise AtriumHTTPError(
                422,
                f"The {label} part is not valid JSON: {exc}.",
                reason="invalid_record",
                errors=[{"msg": str(exc)}],
            ) from exc
    else:
        record = raw
    if not isinstance(record, dict):
        raise AtriumHTTPError(
            422,
            f"The {label} part is a JSON {type(record).__name__}, not an object.",
            reason="invalid_record",
        )
    version = record.get("schema_version")
    if version is not None:
        import atrium_document  # lazily: see the module docstring

        supported = int(str(atrium_document.SCHEMA_VERSION).split(".")[0])
        try:
            major = int(str(version).split(".")[0])
        except ValueError:
            raise AtriumHTTPError(
                422,
                f"The {label} part has schema_version {version!r}, which is not a version.",
                reason="invalid_record",
            ) from None
        if major > supported:
            raise AtriumHTTPError(
                422,
                f"The {label} part has schema_version {version}; this tool reads up to {supported}.x.",
                reason="invalid_record",
            )
    return record


# ──────────────────────────────────────────────────────────────────────────────
# The typed contract (§4.8; atrium-project#32 round 2, items 1 and 3)
# ──────────────────────────────────────────────────────────────────────────────
#
# These models DOCUMENT the responses; they do not filter them. Every route declares
# `response_model=None, responses={200: {"model": ...}}`, so the bytes a service sends are
# exactly what its handler builds, and the per-repo contract tests validate real responses
# against the PUBLISHED schema (atrium_openapi.validate_response). Their docstrings and
# field descriptions are published in every spec, so they are written for the client.
#
# Rules, from the 2026-09-28 design review (agent_dev_logs/plans/32.plan.md):
#   * a field the handler always emits has NO default, so it is required (nullable where it
#     can be null): oasdiff rates removing a required response property as a breaking change,
#     an optional one only after atrium_openapi.py raises that rule;
#   * no enums in responses: a value is an open string whose known values are in its
#     description, so a new value never breaks a client generated from an older spec;
#   * every model allows extra members, so a later additive field is not a breaking change.


class LimitRef(BaseModel):
    """Which limit a `limit_exceeded` refusal is about, and by how much."""

    model_config = ConfigDict(extra="allow")

    key: str = Field(description="The `/info` `limits` key of the limit.")
    env: Optional[str] = Field(description="The environment variable that sets it; null for a derived limit.")
    value: Optional[float] = Field(description="The limit's effective value.")
    observed: Optional[float] = Field(description="What the input measured; null when it is not known exactly.")
    unit: str = Field(description="The unit of `value` and `observed`.")


class ErrorBody(BaseModel):
    """The body of every error a service returns (§4.4)."""

    model_config = ConfigDict(extra="allow")

    status: int = Field(description="The HTTP status, as an integer.")
    reason: Optional[str] = Field(
        description=(
            "A registered reason code, listed with its statuses in the spec's `x-atrium-reason-codes`, or null "
            "when no code is registered for the cause. An open string: a release may add a code, but never "
            "renames or removes one."
        )
    )
    detail: str = Field(description="A human-readable message.")
    limit: Optional[LimitRef] = Field(None, description="Only with `limit_exceeded`: the limit and the input's size.")
    errors: Optional[List[Any]] = Field(None, description="The request-validation problems, or a structured detail.")
    accepted: Optional[List[str]] = Field(
        None, description="Only with `unsupported_media_type`: the file extensions or media types the endpoint reads."
    )
    cause: Optional[str] = Field(
        None, description="A finer, unregistered cause (for example a reader's code). Informational; it may change."
    )


class LimitNote(BaseModel):
    """One limit that shaped a result without refusing it (`limits_applied`, §4.5)."""

    model_config = ConfigDict(extra="allow")

    limit: str = Field(description="The `/info` `limits` key.")
    value: Any = Field(description="The limit's value when it applied.")
    effect: str = Field(description="What it did: `sampled`, `split`, `trimmed`, `skipped` or `stopped`.")
    count: int = Field(description="How many times it applied.")
    detail: str = Field(description="A human-readable note.")


class LimitMeta(BaseModel):
    """Where one limit comes from (`/info` `limits_meta`)."""

    model_config = ConfigDict(extra="allow")

    env: Optional[str] = Field(description="The environment variable that sets it; null for a derived limit.")
    unit: Optional[str] = Field(description="The unit of its value.")
    source: str = Field(description="Where the current value came from: `env`, `config`, `default` or `derived`.")
    default: Optional[float] = Field(None, description="The built-in default (not for a derived limit).")
    derived_from: Optional[List[str]] = Field(
        None, description="For a derived limit: the settings it is computed from."
    )
    zero_means_unlimited: Optional[bool] = Field(None, description="True when 0 switches the limit off.")


class InfoBase(BaseModel):
    """The `/info` envelope every service returns (§4.1); each service adds its own capabilities."""

    model_config = ConfigDict(extra="allow")

    service: str = Field(description="The tool id, which is the repository name.")
    version: str = Field(description="The tool version (`para_config.txt`), as in the release tag without the `v`.")
    endpoints: List[str] = Field(description="The callable API paths.")
    limits: Dict[str, Optional[float]] = Field(description="Every limit of the service: `{key: effective value}`.")
    limits_meta: Dict[str, LimitMeta] = Field(
        description="For each limit: the variable that sets it, its unit, default, source."
    )
    openapi_sha256: Optional[str] = Field(
        description=(
            "sha256 of this process's OpenAPI document in canonical form, equal to the `openapi.json.sha256` "
            "asset of the release the image was built from."
        )
    )


class HealthBody(BaseModel):
    """`GET /health`: liveness, and with `?deep=true` a check of the backends."""

    model_config = ConfigDict(extra="allow")

    status: str = Field(description="`ok`, or `degraded` (HTTP 503) when the deep check fails or the replica drains.")
    detail: Optional[str] = Field(None, description="Why the deep check failed.")
    in_flight: Optional[int] = Field(None, description="Deep check only: requests being served.")
    draining: Optional[bool] = Field(None, description="Deep check only: the replica is shutting down.")


class ReadyBody(BaseModel):
    """`GET /ready`: readiness for new work."""

    model_config = ConfigDict(extra="allow")

    status: str = Field(description="`ready`; or `starting` or `draining`, both with HTTP 503.")


class CreateAction(BaseModel):
    """The RO-Crate `CreateAction` of this call: its provenance (atrium-project#71 R2).

    One Process Run Crate 0.5 action per call, as nested JSON-LD, built by
    `atrium_rocrate.create_action()`; the same document an archive stores as the run's paradata
    file. It names the tool as its `instrument` (with `version`, its release page as `@id` and its
    authors as `creator`), the image as `containerImage`, the inputs as `object`, what the call
    wrote as `result`, the `agent`, `startTime`/`endTime` and `actionStatus`, and the run's whole
    paradata record as `paradataRecord`. Every service returns it with a successful response; its
    `@id` is the `run_uuid` the call stamped into the record it returned.
    """

    model_config = ConfigDict(extra="allow")

    context: Optional[Any] = Field(
        None, alias="@context", description="The JSON-LD context: RO-Crate 1.2, plus the terms ATRIUM adds."
    )
    id: Optional[str] = Field(
        None,
        alias="@id",
        description="The run's `urn:uuid:...`, equal to `run_uuid` in the blocks the call stamped into the record.",
    )
    type: Optional[str] = Field(None, alias="@type", description="`CreateAction`.")
    name: Optional[str] = Field(None, description="A human-readable name of the run.")
    description: Optional[str] = Field(None, description="What the action describes.")
    instrument: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "The tool: a `SoftwareApplication` with `version`, its release page as `@id`, its repository as "
            "`url` and its authors as `creator`."
        ),
    )
    containerImage: Optional[Dict[str, Any]] = Field(
        None,
        description="The image the service ran as (`ATRIUM_RUNNER_IMAGE`): `registry`, `name`, `tag`.",
    )
    object_: Optional[List[Dict[str, Any]]] = Field(
        None,
        alias="object",
        description=(
            "The inputs: the uploaded file (`@id` `ni:///sha-256;...`, with `sha256` and `contentSize`) and "
            "`#record`, the record sent with it."
        ),
    )
    result: Optional[List[Dict[str, Any]]] = Field(
        None, description="The outputs: the record blocks the call wrote (`#block-<name>`) and any output file."
    )
    agent: Optional[Dict[str, Any]] = Field(
        None, description="The organisation operating the service (`ATRIUM_RUN_AGENT`); absent when it is not set."
    )
    startTime: Optional[str] = Field(None, description="When the run started: `YYYY-MM-DDTHH:MM:SS+00:00`.")
    endTime: Optional[str] = Field(None, description="When the run ended: `YYYY-MM-DDTHH:MM:SS+00:00`.")
    actionStatus: Optional[str] = Field(
        None,
        description="`http://schema.org/CompletedActionStatus` or `http://schema.org/FailedActionStatus`.",
    )
    error: Optional[str] = Field(None, description="Why the run failed, with `FailedActionStatus`.")
    paradataRecord: Optional[Dict[str, Any]] = Field(
        None, description="The run's atrium_paradata record (schema 2.0), as the tool wrote it."
    )

    # No return annotation on purpose: the published schema stays the fields above.
    @model_serializer(mode="wrap")
    def _omit_absent_members(self, handler):  # noqa: ANN001, ANN202
        """A JSON-LD action states what it knows: a member it lacks is absent, never `null`.

        A route with a `response_model` (page-classification's `/predict_image`) would otherwise
        send every unset member as `null`, an archive would store them, and `agent: null` reads
        like a claim.
        """
        return {key: value for key, value in handler(self).items() if value is not None}


class AtriumDocument(RootModel[Dict[str, Any]]):
    """An ATRIUM document record (`atrium_document.schema.json`)."""

    model_config = ConfigDict(json_schema_extra={"x-atrium-record": True})


#: The component the record schema is published under, and the prefix of its hoisted `$defs`.
RECORD_COMPONENT = "AtriumDocument"

#: What each error status means to a client (§4.4); the descriptions of :func:`error_responses`.
_ERROR_DESCRIPTIONS: Dict[int, str] = {
    404: "Not found.",
    409: "The resource is not in a state that allows this request.",
    413: "The request content is over a limit (`reason: limit_exceeded`). Split the input or raise the limit.",
    415: "Unsupported media type (`reason: unsupported_media_type`, with `accepted`). Do not retry unchanged.",
    422: (
        "Unusable or invalid input: request validation (`errors`), a bad parameter, a record that cannot be "
        "opened (`reason: invalid_record`), or a parameter or per-input budget over its limit "
        "(`reason: limit_exceeded`). Do not retry unchanged."
    ),
    429: "Busy (`reason: busy`): every processing slot or queue place is taken. Retry after `Retry-After` seconds.",
    500: "Processing failed. Report it; do not retry blindly.",
    501: "This deployment cannot read the input: a reader's optional dependency is not installed.",
    502: "An upstream service failed. Retry three times with backoff.",
    503: "Not ready, warming up or draining. Retry three times with backoff.",
    504: "A time limit that depends on an upstream service expired (`reason: limit_exceeded`). Retry with backoff.",
}


def error_responses(*statuses: int) -> Dict[int, Dict[str, Any]]:
    """The OpenAPI `responses` entries of the given error statuses, each with :class:`ErrorBody`.

    Every service passes ``FastAPI(responses=error_responses(422, 500))``, which puts the
    error body on every route and, because a 422 is declared, stops FastAPI documenting its
    own ``HTTPValidationError`` (which is not the body the services send). Each route adds
    the statuses it can refuse with. A 429 also documents its ``Retry-After`` header.

    The dicts are built fresh on every call: FastAPI copies ``responses`` shallowly, so a
    shared nested dict would be mutated across routes.
    """
    out: Dict[int, Dict[str, Any]] = {}
    for status in statuses:
        entry: Dict[str, Any] = {
            "model": ErrorBody,
            "description": _ERROR_DESCRIPTIONS.get(int(status), _status_phrase(int(status))),
        }
        if int(status) == 429:
            entry["headers"] = {
                "Retry-After": {"description": "Seconds to wait before retrying.", "schema": {"type": "integer"}}
            }
        out[int(status)] = entry
    return out


def operation_id(route: Any) -> str:
    """``generate_unique_id_function`` for every service: the operationId is the route's function name.

    Generated clients name their methods after the operationId, so it must not depend on the
    path or the method (FastAPI's default, e.g. ``predict_image_predict_image_post``). A
    renamed handler is a renamed client method: ``atrium_openapi.py compare`` treats a removed
    operationId as a breaking change.
    """
    return route.name


def attach_openapi_contract(app: FastAPI, service: str, previous: Optional[str] = None) -> None:
    """Finish the OpenAPI document ``app`` generates, the one committed as ``service/openapi.json``.

    Wraps ``app.openapi`` once; the result is cached by FastAPI as usual. On top of what
    FastAPI generates it:

    * adds ``info.x-atrium-service`` (the tool id) and ``x-atrium-reason-codes``
      (``{code: {description, statuses}}``, the registry of :data:`REASON_CODES`);
    * with ``previous`` (a renamed service, atrium-project#72: the id its last release
      published), adds ``info.x-atrium-service-previous``. ``atrium_openapi.py compare`` accepts
      the changed id only when this declaration equals the baseline's; keep it for as long as
      the committed spec should document the rename (it is harmless once the baseline carries
      the new id);
    * replaces the ``AtriumDocument`` component with the vendored
      ``atrium_document.schema.json`` (its ``$defs`` hoisted as ``AtriumDocument_<name>``,
      ``$schema`` and ``$id`` dropped), and records ``x-atrium-record-schema``
      (``schema_version`` and the sha256 of the schema), which ``atrium_openapi.py compare``
      checks instead of diffing the record schema that the schema freeze already guards;
    * drops the app-wide 422 from operations that take no parameters and no body.
    """
    original = app.openapi

    def openapi() -> Dict[str, Any]:
        if app.openapi_schema is None:
            _finish_openapi(original(), service, previous)
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]


def _finish_openapi(spec: Dict[str, Any], service: str, previous: Optional[str] = None) -> None:
    info = spec.setdefault("info", {})
    info["x-atrium-service"] = service
    if previous:
        info["x-atrium-service-previous"] = previous
    spec["x-atrium-reason-codes"] = {
        code: {"description": REASON_CODES[code], "statuses": list(REASON_STATUSES[code])} for code in REASON_CODES
    }
    schemas = spec.get("components", {}).get("schemas", {})
    if RECORD_COMPONENT in schemas:
        import atrium_document  # lazily: see the module docstring
        import atrium_openapi  # lazily: see the module docstring

        record = atrium_document.load_schema()
        spec["x-atrium-record-schema"] = {
            "schema_version": atrium_document.SCHEMA_VERSION,
            "sha256": atrium_openapi.digest(record),
        }
        body, hoisted = _hoist_record_schema(record)
        schemas[RECORD_COMPONENT] = body
        schemas.update(hoisted)
    for operations in spec.get("paths", {}).values():
        for operation in operations.values():
            if isinstance(operation, dict) and not operation.get("parameters") and "requestBody" not in operation:
                operation.get("responses", {}).pop("422", None)


def _hoist_record_schema(record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    body = {key: value for key, value in record.items() if key not in ("$schema", "$id", "$defs")}
    defs = record.get("$defs", {})
    names = {f"#/$defs/{name}": f"#/components/schemas/{RECORD_COMPONENT}_{name}" for name in defs}

    def rewrite(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: (names.get(v, v) if k == "$ref" and isinstance(v, str) else rewrite(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [rewrite(item) for item in node]
        return node

    return rewrite(body), {f"{RECORD_COMPONENT}_{name}": rewrite(schema) for name, schema in defs.items()}


def openapi_digest(app: FastAPI) -> Optional[str]:
    """The sha256 of ``app``'s OpenAPI document in canonical form (``atrium_openapi.digest``).

    ``None`` if the document cannot be built: ``/info`` must answer regardless, and the
    per-repo contract test fails on the ``None`` instead.
    """
    try:
        import atrium_openapi  # lazily: see the module docstring

        return atrium_openapi.digest(app.openapi())
    except Exception:  # /info must never fail because the spec cannot be built
        return None


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

    Both routes declare their bodies (:class:`HealthBody`, :class:`ReadyBody`) for 200 and
    503 in the spec (atrium-project#32 round 2); what they send is unchanged.
    """

    @app.get(
        "/health",
        response_model=None,
        responses={
            200: {"model": HealthBody, "description": "Alive; with `?deep=true`, the backends answer too."},
            503: {"model": HealthBody, "description": "`degraded`: the deep check failed, or the replica drains."},
        },
    )
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

        @app.get(
            "/ready",
            response_model=None,
            responses={
                200: {"model": ReadyBody, "description": "`ready`: route new work here."},
                503: {"model": ReadyBody, "description": "`starting` or `draining`: route new work elsewhere."},
            },
        )
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

    A blank value counts as unset. A malformed or negative one raises
    ``atrium_limits.LimitConfigError`` naming the variable (atrium-project#53): it used
    to be ignored silently, so a typo ran the service on the default with nothing to say
    so. ``atrium_limits.upload_limit()`` is the same rule as a ``LimitSpec``, for a
    repo's ``LimitSet``; this function stays for the services that have not adopted one.
    """
    raw_mb = os.getenv("MAX_UPLOAD_MB")
    if raw_mb is not None and raw_mb.strip():
        return _strict_mb(raw_mb, "MAX_UPLOAD_MB", 1.0)
    legacy_bytes = os.getenv("MAX_UPLOAD_BYTES")
    if legacy_bytes is not None and legacy_bytes.strip():
        return _strict_mb(legacy_bytes, "MAX_UPLOAD_BYTES", 1024 * 1024)
    return float(default_mb)


def _strict_mb(raw: str, name: str, divisor: float) -> float:
    try:
        value = float(raw.strip())
    except ValueError:
        value = float("nan")
    if not (value >= 0 and value != float("inf")):
        import atrium_limits  # lazily: see the module docstring

        raise atrium_limits.LimitConfigError(
            f"The environment variable {name} is {raw.strip()!r}; it must be a non-negative number."
        )
    return value / divisor


def allowed_origins(default: str = "*") -> List[str]:
    """Parse ``ALLOWED_ORIGINS`` (CSV) into a list; default single wildcard (§4.5)."""
    return [o.strip() for o in os.getenv("ALLOWED_ORIGINS", default).split(",") if o.strip()]


#: Response headers a browser client may read (CORS ``expose_headers``): the ``busy``
#: refusal's ``Retry-After``, and the translator's limits-applied summary (#53).
EXPOSED_HEADERS = ("Retry-After", "X-Atrium-Limits-Applied")


def add_cors(
    app: FastAPI,
    methods: Optional[Iterable[str]] = None,
    default_origins: str = "*",
    expose_headers: Iterable[str] = EXPOSED_HEADERS,
) -> None:
    """Attach the standard CORS middleware (§4.5).

    Origins come from ``ALLOWED_ORIGINS`` (CSV, default ``*``). Credentials are enabled
    only when the origin list is not the bare ``*`` wildcard — browsers reject the
    wildcard+credentials combination. ``expose_headers`` lets a browser read the headers
    in :data:`EXPOSED_HEADERS`, which CORS otherwise hides from scripts.
    """
    origins = allowed_origins(default_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=list(methods) if methods else ["*"],
        allow_headers=["*"],
        expose_headers=list(expose_headers),
    )
