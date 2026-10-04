"""
service/api.py — FastAPI service for ATRIUM page classification.

The typed contract (atrium-project#32 round 2). Every route declares its response model
and its error statuses, so the committed ``service/openapi.json`` — attached to every
release, and what the AMČR pipeline generates its clients from — types every field.
``/predict_image`` keeps its ``response_model``; ``/predict_document`` and ``/info``
DOCUMENT theirs (``response_model=None``), and ``tests/test_api_contract.py`` validates real
responses against the published schema. Refusals carry registered reasons: a wrong media
type is 415 ``unsupported_media_type``, a record that cannot be opened is 422
``invalid_record``. Regenerate the spec after an API change (the model manager is stubbed,
as ``tests/openapi_contract_data.py`` does, so torch is not needed)::

    python atrium_openapi.py export --app service.api:app --out service/openapi.json \
        --prepare tests.openapi_contract_data:prepare
"""

import asyncio
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

# [FIX]: Use a relative import to support pytest running from the repo root,
# with a fallback for direct script execution.
try:
    from .inference import manager
except ImportError:
    from inference import manager

# The PDF engine (atrium-project#72 D.1): pypdfium2, called only through this torch-free module,
# which data_scripts/compare_pdf_rasterisers.py renders with too.
try:
    from .pdf_render import LOCK as _PDFIUM_LOCK
    from .pdf_render import open_document, render_page, render_scale
except ImportError:
    from pdf_render import LOCK as _PDFIUM_LOCK
    from pdf_render import open_document, render_page, render_scale

# Accretion contract rule 1 — optional `document_json` part in, updated record out
# (atrium-project#10, J2). Kept in its own torch-free module so the accretion is testable in
# the fast lane; see service/document_json.py's docstring for why that matters here.
try:
    from .document_json import (
        baseline_page_labels,
        build_document_record,
        doc_id_for_document,
        doc_id_for_image,
        expand_page_selection,
        parse_page_selection,
        record_page_keys,
    )
except ImportError:
    from document_json import (
        baseline_page_labels,
        build_document_record,
        doc_id_for_document,
        doc_id_for_image,
        expand_page_selection,
        parse_page_selection,
        record_page_keys,
    )

# Shared ATRIUM meta-contract helpers (§4). Byte-identical across every service,
# enforced by para-drift.reusable.yml — same relative-vs-bare import dance.
try:
    from .atrium_service import (
        AtriumDocument,
        AtriumHTTPError,
        CreateAction,
        InfoBase,
        LimitNote,
        ServiceState,
        add_cors,
        attach_error_handlers,
        attach_health,
        attach_inflight_middleware,
        attach_openapi_contract,
        build_info,
        error_responses,
        operation_id,
        parse_record_part,
        read_tool_version,
        read_upload_bounded,
        serve_lifecycle,
    )
except ImportError:
    from atrium_service import (
        AtriumDocument,
        AtriumHTTPError,
        CreateAction,
        InfoBase,
        LimitNote,
        ServiceState,
        add_cors,
        attach_error_handlers,
        attach_health,
        attach_inflight_middleware,
        attach_openapi_contract,
        build_info,
        error_responses,
        operation_id,
        parse_record_part,
        read_tool_version,
        read_upload_bounded,
        serve_lifecycle,
    )

# Every limit this service has, declared once (atrium-project#53, factor III). The repo
# root is on sys.path by now: service/inference.py, imported above, puts it there.
import tool_limits  # noqa: E402
from atrium_limits import LimitExceeded  # noqa: E402
from tool_limits import LIMITS, MAX_IMAGE_PIXELS, MAX_PDF_PAGES, MAX_UPLOAD  # noqa: E402

logger = logging.getLogger(__name__)

#: The tool id (/info `service`, the spec's `x-atrium-service`): the repository name.
SERVICE = "atrium-page-classification"

# Import-time snapshots, kept because tests and clients import them. The service itself
# reads each limit per request from tool_limits (atrium_limits reads the environment on
# every call), so these are what the limits WERE when the module was imported.
MAX_UPLOAD_MB = MAX_UPLOAD.get()
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

# The service checks the pixel count itself, against MAX_IMAGE_PIXELS, before anything is
# decoded (_check_image_pixels). Pillow's own process-wide guard is switched off here
# because it made the effective limit depend on request history: utils.py raises it to
# 4.22 G px when it is first imported, which happens lazily on the first /predict_image,
# so the first request of a process ran with Pillow's default and every later one did not.
Image.MAX_IMAGE_PIXELS = None

#: The PDF rendering resolution's value at import, kept for the callers and tests that read
#: it (it was a constant here). A setting since atrium-project#53: requests read
#: tool_limits.PDF_RENDER_DPI, so it is reported in /info and can be changed.
PDF_RENDER_DPI = tool_limits.PDF_RENDER_DPI.get()

#: `topn` bounds: at most one label per category. Out of range is a request-validation 422,
#: not a silent cap (version="all" used to return all categories for any larger value) or a
#: torch error surfacing as a 500 (a single version). model_registry is torch-free.
from model_registry import CATEGORIES as _CATEGORIES  # noqa: E402
from model_registry import resolve_base_model  # noqa: E402

_TOPN_MAX = len(_CATEGORIES)


#: Readiness/draining/in-flight state for the §4.6 disposability contract (issue #55).
_state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models on startup. Off the event loop: warmup loads several torch
    # models, and doing that inline would block the loop through the whole startup
    # (harmless before traffic, but it also delays the first /ready answer a
    # startupProbe is waiting on).
    logger.info("Warming up models...")
    await asyncio.to_thread(manager.warmup)
    _state.warm = True
    # issue #55: composes with the warmup above rather than replacing it. Flips /ready
    # to 503 on SIGTERM and waits for in-flight classification to finish before exit.
    async with serve_lifecycle(_state):
        yield
    # Cleanup resources on shutdown if necessary
    logger.info("Shutting down API service...")


app = FastAPI(
    title="ATRIUM Page Classification API",
    version=read_tool_version(Path(__file__).resolve().parent),
    description="API for classifying historical document page images.",
    lifespan=lifespan,
    # The typed contract (atrium-project#32 round 2): every route documents the §4.4 error
    # body for 422 and 500 (and FastAPI's own 422 body, which is not what is sent, goes);
    # operationIds are the handler names; the spec never depends on a root_path.
    responses=error_responses(422, 500),
    generate_unique_id_function=operation_id,
    root_path_in_servers=False,
)
attach_inflight_middleware(app, _state)
# §4.4 error body {status, reason, detail} for every error (atrium-project#32 item 2, #53).
attach_error_handlers(app)
# The published spec: reason registry, record schema, service id (atrium-project#32 item 3).
attach_openapi_contract(app, SERVICE)

# CORS — standard §4.5 configuration (ALLOWED_ORIGINS CSV, default "*").
add_cors(app, methods=["GET", "POST"])


def _deep_health() -> str | None:
    """Deep readiness (§4.1): at least one classification model version is loaded."""
    try:
        if not manager.available_versions:
            return "no model versions available"
    except Exception as exc:
        return f"model manager not ready: {exc}"
    return None


attach_health(app, deep_check=_deep_health, state=_state)


def _check_image_pixels(width: int, height: int, what: str) -> None:
    """Refuse an image over MAX_IMAGE_PIXELS (413 ``limit_exceeded``) before decoding it."""
    MAX_IMAGE_PIXELS.check(
        width * height,
        detail=(
            f"{what} is {width} x {height} = {width * height} pixels; the limit is "
            f"{MAX_IMAGE_PIXELS.get()} (MAX_IMAGE_PIXELS)."
        ),
    )


def _check_version(version: str) -> None:
    """Refuse a `version` no model revision matches (422), before anything is read or run.

    `all` is the ensemble; any other value must resolve to a base model by the rule the model
    manager loads by (model_registry.resolve_base_model). Such a value used to reach the loader
    and come back as a 500 "Error processing image." with the classifier to blame
    (atrium-project#32 round 2). A revision that resolves but whose weights cannot be loaded is
    still the manager's failure, a 500.
    """
    if version != "all" and resolve_base_model(version) is None:
        known = ", ".join(["all", *manager.available_versions])
        raise HTTPException(
            status_code=422, detail=f"Unknown model version {version!r}; the published ones are {known}."
        )


def _open_image(content: bytes) -> Image.Image:
    """Open the uploaded image (its header only), or refuse it: 422 for bytes that are not a
    readable image (atrium-project#32 round 2; the blanket 500 before)."""
    try:
        return Image.open(io.BytesIO(content))
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(status_code=422, detail=f"The upload is not a readable image: {exc}") from exc


def _decode_image(image: Image.Image) -> Image.Image:
    """Decode the pixels as RGB, or refuse a damaged or truncated image with a 422."""
    try:
        return image.convert("RGB")
    except (OSError, SyntaxError) as exc:
        raise HTTPException(status_code=422, detail=f"The upload is not a readable image: {exc}") from exc


def _open_pdf(content: bytes):
    """Open the uploaded PDF, or refuse it with a 422 when it cannot be read. Hold ``_PDFIUM_LOCK``.

    ``PdfiumError`` (a RuntimeError) for bytes that are not a PDF, a damaged one or a locked
    one is the caller's input, not our failure — the blanket 500 "Error processing document."
    before atrium-project#32 round 2. The engine is pypdfium2 since atrium-project#72 D.1
    (``service/pdf_render.py``); PyMuPDF (AGPL-3.0, declared nowhere) did this before.
    """
    import pypdfium2 as pdfium

    try:
        return open_document(content)
    except pdfium.PdfiumError as exc:
        raise HTTPException(status_code=422, detail=f"The upload is not a readable PDF: {exc}") from exc


def _classify_pdf_pages(
    content: bytes, version: str, topn: int, selection: Optional[List[Tuple[int, int]]] = None
) -> List[Dict[str, Any]]:
    """Rasterise the pages of a PDF and classify them — the blocking body of
    ``POST /predict_document``, extracted so it runs in ONE worker thread (issue #55).

    Every page, or only those `selection` names (the `pages` field, already parsed; a page past
    the end is a 422, checked here because only here is the page count known).

    Raises ``atrium_limits.LimitExceeded`` (413) when more pages would be classified than
    MAX_PDF_PAGES — the whole PDF, or the selection — and for a
    page that would render over MAX_IMAGE_PIXELS at PDF_RENDER_DPI — the size is computed
    from the page's own dimensions before it is rendered, so a small file declaring a huge
    page cannot make the service rasterise gigabytes. Raised in the worker thread, it reaches
    the error handler the same as on the loop.

    PDFium is not thread-safe, so every call into it holds ``_PDFIUM_LOCK``
    (``service/pdf_render.py``): pages are rendered one at a time under the lock and
    classified outside it, so a long model run never holds PDFium from another request.

    A page the model could not classify fails the request with a 500 naming the page
    (atrium-project#32 round 2): its `{"error": ...}` used to be returned in place of the
    page's predictions, inside a 200.
    """
    with _PDFIUM_LOCK:
        pdf_document = _open_pdf(content)
    try:
        with _PDFIUM_LOCK:
            page_count = len(pdf_document)
        try:
            numbers = expand_page_selection(selection, page_count)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"pages: {exc}.") from exc
        if selection is None:
            detail = f"PDF has too many pages: {page_count}. Limit is {MAX_PDF_PAGES.get()} (MAX_PDF_PAGES)."
        else:
            detail = (
                f"Too many pages selected: {len(numbers)} of the PDF's {page_count}. "
                f"Limit is {MAX_PDF_PAGES.get()} (MAX_PDF_PAGES)."
            )
        MAX_PDF_PAGES.check(len(numbers), detail=detail)

        dpi = tool_limits.PDF_RENDER_DPI.get()
        scale = render_scale(dpi)
        page_results: List[Dict[str, Any]] = []
        for number in numbers:
            what = f"Page {number} at {dpi} dpi"
            with _PDFIUM_LOCK:
                img = render_page(pdf_document, number - 1, scale, lambda w, h: _check_image_pixels(w, h, what))

            predictions = manager.predict(img, version=version, topn=topn)
            if isinstance(predictions, dict) and "error" in predictions:
                raise HTTPException(status_code=500, detail=f"Page {number}: {predictions['error']}")
            page_results.append({"page": number, "predictions": predictions})
        return page_results
    finally:
        with _PDFIUM_LOCK:
            pdf_document.close()


def _refuse_if_draining() -> None:
    """Reject NEW work once a shutdown signal has arrived (issue #55).

    /ready has already flipped to 503 by this point, but a request that was accepted
    before the orchestrator noticed can still reach a handler — answering 503 here
    keeps the set of requests the drain has to wait for bounded, and tells the client
    to retry against a live replica instead of failing mid-inference.
    """
    if _state.draining:
        raise HTTPException(status_code=503, detail="Service is shutting down; retry against a live replica.")


# Mount frontend
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ── the typed contract (atrium-project#32 round 2) ──────────────────────────────────────────
# `/predict_image` returns an ImageResponse through `response_model`, so every field below is
# sent, null or not: those without a default are required (nullable where null). The other two
# models DOCUMENT their responses (`response_model=None`): a field the handler always sends has
# no default, one it sends only sometimes defaults to None. Descriptions are published in
# service/openapi.json, so they are written for the client. Labels are open strings: a new
# category must never break a client generated from an older spec.


class PredictionResult(BaseModel):
    """One category and its score."""

    label: str = Field(description="A page category (`/info` `categories`), e.g. `TEXT`, `DRAW`, `PHOTO_L`.")
    score: float = Field(description="Its score, from 0 to 1; the ensemble's is the mean of the models'.")


class ImageResponse(BaseModel):
    """`/predict_image`: the categories of one page image, and its record when one was asked for."""

    type: str = Field(description="`image`.")
    predictions: List[PredictionResult] = Field(description="The `topn` best categories, best first.")
    #: The limits that shaped this result without refusing it (atrium-project#53,
    #: docs/paradata_schema.md `limits_applied`). None of this service's limits does that
    #: today — each one refuses — so it is `[]`; it is declared so the field is present in
    #: every service's response, and so `response_model` does not filter it out.
    limits_applied: List[LimitNote] = Field(
        description="Every limit that shaped the result without refusing it; `[]` today (each limit refuses)."
    )
    #: The updated ATRIUM Document JSON, present only when the caller opted into the
    #: accretion flow (uploaded a `document_json` baseline, or asked for `document_json_out`).
    #: `response_model` filters unknown keys, so these have to be declared here or the record
    #: is silently dropped on the way out — which is J2 all over again, one layer down.
    document_json: Optional[AtriumDocument] = Field(
        description=(
            "With a `document_json` baseline or `document_json_out=true`: the record with page-classification's "
            "`page_categories` block and `pages[].category` / `pages[].category_confidence` updated; else null."
        )
    )
    #: Non-None only in Layer D's inherited-defect case: the uploaded baseline did not
    #: validate, so the record was emitted with a warning rather than refused. A field an
    #: automated caller can test, instead of a line it would have to grep the service log for.
    document_json_schema_error: Optional[str] = Field(
        description="Only when the sent baseline did not validate (the record is returned anyway): the error; else null."
    )
    paradata: Optional[CreateAction] = Field(
        description="The run's provenance (atrium-project#67 R2). Not returned yet: always null."
    )


class PagePredictions(BaseModel):
    """The categories of one PDF page."""

    model_config = ConfigDict(extra="allow")

    page: int = Field(description="The page's physical position in the PDF, 1-based.")
    page_label: Optional[str] = Field(
        None,
        description=(
            "Only when the `document_json` baseline has a page row with this `page_index`: that row's `page` "
            "(the PDF page label, e.g. `iv`), the key this page's category is written under in the record."
        ),
    )
    predictions: List[PredictionResult] = Field(description="The `topn` best categories, best first.")


class DocumentResponse(BaseModel):
    """`/predict_document`: the categories of every page of a PDF, and its record when one was asked for."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="`document`.")
    pages: List[PagePredictions] = Field(
        description="One entry per classified page (every page, or the `pages` selection), in page order."
    )
    limits_applied: List[LimitNote] = Field(
        description="Every limit that shaped the result without refusing it; `[]` today (each limit refuses)."
    )
    document_json: Optional[AtriumDocument] = Field(
        None,
        description=(
            "Only with a `document_json` baseline or `document_json_out=true`: the record with "
            "page-classification's `page_categories` block and `pages[]` fields updated for every classified "
            "page, keyed by the baseline's own page labels where its rows carry `page_index`."
        ),
    )
    document_json_schema_error: Optional[str] = Field(
        None, description="Only when the sent baseline did not validate (the record is returned anyway): the error."
    )
    paradata: Optional[CreateAction] = Field(
        None, description="The run's provenance (atrium-project#67 R2). Not returned yet: always absent."
    )


class PcInfo(InfoBase):
    """`/info` of atrium-page-classification."""

    categories: List[str] = Field(description="The page categories a prediction can name.")
    available_models: Dict[str, str] = Field(
        description="The published model revisions and `all` (the ensemble), each with its base model."
    )


#: What the `version` field of both endpoints means.
_VERSION_DESC = (
    "`all` (the default): the ensemble of the published revisions, averaged; or one model revision, "
    "e.g. `v4.3` (`/info` `available_models`). A value no revision matches is refused (422)."
)


# Shared description strings — both endpoints advertise the identical contract, and OpenAPI is
# the only documentation an API consumer reads.
_DOCUMENT_JSON_DESC = (
    "Optional baseline ATRIUM Document JSON (accretion model, docs/document_schema.md / "
    "issue #13). When given, the response's `document_json` carries the record back with only "
    "page-classification's `page_categories` block and `pages[].category` / "
    "`pages[].category_confidence` fields updated — every other tool's block passes through "
    "untouched. A baseline that does not validate against atrium_document.schema.json is still "
    "accepted (rule 6), but the response then also carries `document_json_schema_error`; one that "
    "cannot be opened is refused (422 `invalid_record`). An empty part counts as none."
)
_PAGES_DESC = (
    "Optional: classify only these pages, 1-based physical positions, e.g. `1,3,5-7`. Empty (the default): every "
    "page. A malformed value, or a page past the end of the PDF, is refused (422); MAX_PDF_PAGES counts the pages "
    "classified. The response lists only these pages, and only they are written into the record."
)
_DOCUMENT_JSON_OUT_DESC = (
    "Return a document record even with no baseline uploaded. page-classification is stage 1 "
    "of the pipeline, so it ORIGINATES the record (the E2E smoke passes it "
    "`--document-json-out` alone); without this flag the service could accrete onto someone "
    "else's record but never start one. Mirrors the CLI's `--document-json-out`."
)


async def _read_record_part(document_json: Optional[UploadFile]) -> Optional[bytes]:
    """The baseline part's bytes, read and opened BEFORE any model runs, or None.

    `None` for an absent part and for an empty one: some clients send the multipart field
    with an empty body rather than omitting it. That means "no baseline", not "a baseline
    that is zero bytes long" — taken literally it reaches load_document() and dies on a
    JSONDecodeError. Bounded like the main upload (atrium-project#53): it used to be read
    whole, with no limit at all. A part that cannot be opened (not UTF-8 JSON, not an
    object, a newer `schema_version` major) is refused here, 422 `invalid_record`
    (atrium-project#32 round 2) — it used to be found only after the classification.
    """
    if document_json is None:
        return None
    raw = await read_upload_bounded(document_json, MAX_UPLOAD.get(), "document_json")
    return raw if parse_record_part(raw, "document_json") is not None else None


async def _document_json_part(
    wants_record: bool,
    baseline_bytes: Optional[bytes],
    doc_id: str,
    pages: Sequence[Tuple[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Run this tool's blocks through the CLI's own accretion path, or do nothing.

    Opt-in, like translator's and llm-enrich's services: a caller that sends neither part gets
    the exact response shape it got before, so this is additive on the wire. `wants_record` is
    `document_json_out`, or a `document_json` part sent at all — an empty one included, which
    originates the record as it always did; `baseline_bytes` is what :func:`_read_record_part`
    returned for it.
    """
    if not wants_record:
        return None, None

    try:
        return build_document_record(doc_id, pages, baseline_bytes)
    except ValueError as exc:
        # A baseline the adapter cannot migrate. _read_record_part already refused one that is
        # not JSON or has a newer schema_version major; this is the adapter's own refusal of
        # the rest. Both are the CALLER's payload, so §4.4 says 422, not 500 — and certainly
        # not the endpoint's blanket "Error processing image.", which would send somebody
        # debugging their upload to look at the classifier.
        raise AtriumHTTPError(422, f"Unusable document_json baseline: {exc}", reason="invalid_record") from exc
    except RuntimeError as exc:
        # The adapter's Layer D refusal (D4). Mapped here rather than left to the endpoint's
        # blanket "Error processing image." 500, which would say nothing about why. It stays a
        # 500 and not a 4xx: a record page-classification cannot emit is a defect on THIS side,
        # and the adapter already warns-and-emits instead of raising whenever the invalidity
        # was inherited from the caller's baseline.
        logger.error(f"Document record rejected by its own schema: {exc}")
        raise HTTPException(status_code=500, detail=f"Document record rejected by its own schema: {exc}") from exc


@app.get("/")
def read_root():
    return {"message": "Welcome to the ATRIUM Page Classification API. Use /info for available models."}


@app.get(
    "/info",
    response_model=None,
    responses={200: {"model": PcInfo, "description": "Identity, limits, capabilities."}},
)
def get_info():
    """Return service identity, capabilities, and available model versions (§4.1)."""
    # [FIX]: Removed the hardcoded fallback list.
    # model_registry is the single source of truth.
    from model_registry import CATEGORIES

    model_info = {v: manager.get_model_details(v) for v in manager.available_versions}
    model_info["all"] = manager.get_model_details("all")

    return build_info(
        app,
        service=SERVICE,
        limits=LIMITS,
        categories=CATEGORIES,
        available_models=model_info,
    )


@app.post(
    "/predict_image",
    response_model=ImageResponse,
    responses={
        200: {"description": "The categories, and the record when one was asked for."},
        **error_responses(413, 415, 503),
    },
)
async def predict_image(
    version: str = Form("all", description=_VERSION_DESC),
    topn: int = Form(3, ge=1, le=_TOPN_MAX, description="How many categories to return, best first."),
    file: UploadFile = File(..., description="The page image (any `image/*` type Pillow reads)."),
    document_json: UploadFile = File(
        None, description=_DOCUMENT_JSON_DESC, json_schema_extra={"contentMediaType": "application/json"}
    ),
    document_json_out: bool = Form(False, description=_DOCUMENT_JSON_OUT_DESC),
):
    """Classify a single uploaded image."""
    _refuse_if_draining()
    if not file.content_type or not file.content_type.startswith("image/"):
        # §4.4: a media type this endpoint does not read is 415 `unsupported_media_type` (a
        # 400 before atrium-project#32 round 2), with the accepted type in the body.
        raise AtriumHTTPError(
            415, "Invalid file type. Please upload an image.", reason="unsupported_media_type", accepted=["image/*"]
        )
    _check_version(version)

    content = await read_upload_bounded(file, MAX_UPLOAD.get(), "File")
    baseline_bytes = await _read_record_part(document_json)

    try:
        image = _open_image(content)  # reads the header only
        _check_image_pixels(image.width, image.height, "The image")
        image = _decode_image(image)
        # Off the event loop (issue #55): manager.predict() is synchronous torch
        # inference. Called inline in an `async def`, it blocks the ONLY event loop, so
        # uvicorn's SIGTERM handler — an event-loop callback — could not run until the
        # inference returned, which made --timeout-graceful-shutdown meaningless here.
        predictions = await asyncio.to_thread(manager.predict, image, version=version, topn=topn)
        if isinstance(predictions, dict) and "error" in predictions:
            raise HTTPException(status_code=500, detail=predictions["error"])

        # A single image is ONE PAGE of a document, and which page is carried in its filename
        # — so the doc_id/page split is utils.doc_id_and_page(), the same derivation run.py's
        # -f path uses, or the service would fork the record it is supposed to accrete onto.
        doc_id, page_key = doc_id_for_image(file.filename)
        record, schema_err = await _document_json_part(
            document_json_out or document_json is not None, baseline_bytes, doc_id, [(page_key, predictions)]
        )

        return ImageResponse(
            type="image",
            predictions=predictions,
            limits_applied=[],
            document_json=record,
            document_json_schema_error=schema_err,
            paradata=None,
        )
    except (HTTPException, LimitExceeded):
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image.")


@app.post(
    "/predict_document",
    response_model=None,
    responses={
        200: {
            "model": DocumentResponse,
            "description": "The categories of every page, and the record when one was asked for.",
        },
        **error_responses(413, 415, 503),
    },
)
async def predict_document(
    version: str = Form("all", description=_VERSION_DESC),
    topn: int = Form(3, ge=1, le=_TOPN_MAX, description="How many categories to return per page, best first."),
    file: UploadFile = File(..., description="The PDF (`application/pdf`)."),
    document_json: UploadFile = File(
        None, description=_DOCUMENT_JSON_DESC, json_schema_extra={"contentMediaType": "application/json"}
    ),
    document_json_out: bool = Form(False, description=_DOCUMENT_JSON_OUT_DESC),
    pages: Optional[str] = Form(None, description=_PAGES_DESC),
):
    """Extracts pages from a PDF and classifies each page (or the `pages` selection)."""
    _refuse_if_draining()
    if not file.content_type or file.content_type != "application/pdf":
        # §4.4: 415 `unsupported_media_type` (a 400 before atrium-project#32 round 2).
        raise AtriumHTTPError(
            415,
            "Invalid file type. Please upload a PDF.",
            reason="unsupported_media_type",
            accepted=["application/pdf"],
        )
    _check_version(version)
    try:
        selection = parse_page_selection(pages)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"pages: {exc}.") from exc

    content = await read_upload_bounded(file, MAX_UPLOAD.get(), "File")
    baseline_bytes = await _read_record_part(document_json)

    try:
        # One thread hop for the WHOLE loop, not one per page (issue #55): this is up to
        # MAX_PDF_PAGES sequential torch inferences, and running it inline in an
        # `async def` blocked the event loop — and therefore uvicorn's SIGTERM handler —
        # for the entire duration. See _classify_pdf_pages below.
        page_results = await asyncio.to_thread(_classify_pdf_pages, content, version, topn, selection)

        # A PDF is a whole document: the pages are its own 1..N, so no filename page-split here —
        # just the canonical doc_id. A baseline that already names its pages (a born-digital record:
        # PDF page labels, with the position in `page_index`) gets each category under its own
        # label, never under a position that may be another page's label
        # (atrium-digital-convert#2).
        labels = baseline_page_labels(baseline_bytes)
        keys = record_page_keys([r["page"] for r in page_results], labels)
        for result in page_results:
            if result["page"] in labels:
                result["page_label"] = labels[result["page"]]
        record, schema_err = await _document_json_part(
            document_json_out or document_json is not None,
            baseline_bytes,
            doc_id_for_document(file.filename),
            [(keys[r["page"]], r["predictions"]) for r in page_results if keys[r["page"]] is not None],
        )

        # limits_applied: see ImageResponse — present in every response, [] here today.
        response: Dict[str, Any] = {"type": "document", "pages": page_results, "limits_applied": []}
        if record is not None:
            response["document_json"] = record
        if schema_err:
            response["document_json_schema_error"] = schema_err
        return response
    except (HTTPException, LimitExceeded):
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="Error processing document.")


if __name__ == "__main__":
    import logging
    import os
    import sys

    import uvicorn

    # (12-factor XI) Logs are an event stream: emit to stdout and let the supervisor
    # route them. The library modules only getLogger(); this is the one place allowed
    # to configure handlers. The format string is alto-postprocess's, verbatim, in all
    # five services — a partner tailing five logs wants one shape, and format drift is
    # never fixed later. (issue #61)
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    # (12-factor VII) The service exports itself by binding a port, and which port is
    # configuration. This was baked into an exec-form ENTRYPOINT array, where no shell
    # exists to expand a variable even if one is set — while the reference manifest we
    # hand ARÚP/ARÚB (atrium-project docs/templates/k8s/atrium-service.deployment.yaml)
    # declares `env: PORT` and service/healthcheck.py already reads it. Setting PORT
    # therefore moved the health PROBE and not the listener, so the container reported
    # unhealthy forever rather than simply ignoring the knob. (issue #58)
    reload = os.getenv("RELOAD", "false").strip().lower() in ("true", "1", "yes", "on")

    # uvicorn needs an IMPORT STRING to respawn workers on reload; everywhere else the
    # app OBJECT is correct and strictly better. Passing a string under the container
    # entrypoint (`python -m service.api`) re-imports this module under its real name
    # while it is already running as __main__: the whole body executes twice, and the
    # copy uvicorn serves is not the one __main__ built. __spec__ is None under a direct
    # `python api.py` from service/ (service/README.md's documented start), where no
    # import string resolves anyway — so reload degrades to a uvicorn warning there
    # instead of silently pretending to be on.
    _app_ref = f"{__spec__.name}:app" if reload and __spec__ is not None else app

    uvicorn.run(
        _app_ref,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=reload,
        # (12-factor IX) Disposability: this is the `--timeout-graceful-shutdown 20`
        # that moved off the ENTRYPOINT line when the port became configurable. It
        # bounds uvicorn's wait for in-flight requests; serve_lifecycle() adds its own
        # drain on top, and docs/k8s_deployment.md in the hub carries the full grace
        # budget the two have to fit inside. (issue #55)
        timeout_graceful_shutdown=int(os.getenv("GRACEFUL_SHUTDOWN_S", "20")),
    )
