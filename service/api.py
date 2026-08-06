import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# [FIX]: Use a relative import to support pytest running from the repo root,
# with a fallback for direct script execution.
try:
    from .inference import manager
except ImportError:
    from inference import manager

# Accretion contract rule 1 — optional `document_json` part in, updated record out
# (atrium-project#10, J2). Kept in its own torch-free module so the accretion is testable in
# the fast lane; see service/document_json.py's docstring for why that matters here.
try:
    from .document_json import build_document_record, doc_id_for_document, doc_id_for_image
except ImportError:
    from document_json import build_document_record, doc_id_for_document, doc_id_for_image

# Shared ATRIUM meta-contract helpers (§4). Byte-identical across every service,
# enforced by para-drift.reusable.yml — same relative-vs-bare import dance.
try:
    from .atrium_service import add_cors, attach_health, build_info, read_tool_version, resolve_max_upload_mb
except ImportError:
    from atrium_service import add_cors, attach_health, build_info, read_tool_version, resolve_max_upload_mb

logger = logging.getLogger(__name__)

# Canonical upload limit (§4.5): MAX_UPLOAD_MB, with a MAX_UPLOAD_BYTES fallback.
MAX_UPLOAD_MB = resolve_max_upload_mb(10)
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)  # retained: imported by tests/clients
MAX_PDF_PAGES = 50


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models on startup
    logger.info("Warming up models...")
    manager.warmup()
    yield
    # Cleanup resources on shutdown if necessary
    logger.info("Shutting down API service...")


app = FastAPI(
    title="ATRIUM Page Classification API",
    version=read_tool_version(Path(__file__).resolve().parent),
    description="API for classifying historical document page images.",
    lifespan=lifespan,
)

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


attach_health(app, deep_check=_deep_health)

# Mount frontend
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


class PredictionResult(BaseModel):
    label: str
    score: float


class ImageResponse(BaseModel):
    type: str
    predictions: List[PredictionResult]
    #: The updated ATRIUM Document JSON, present only when the caller opted into the
    #: accretion flow (uploaded a `document_json` baseline, or asked for `document_json_out`).
    #: `response_model` filters unknown keys, so these have to be declared here or the record
    #: is silently dropped on the way out — which is J2 all over again, one layer down.
    document_json: Optional[Dict[str, Any]] = None
    #: Non-None only in Layer D's inherited-defect case: the uploaded baseline did not
    #: validate, so the record was emitted with a warning rather than refused. A field an
    #: automated caller can test, instead of a line it would have to grep the service log for.
    document_json_schema_error: Optional[str] = None


# Shared description strings — both endpoints advertise the identical contract, and OpenAPI is
# the only documentation an API consumer reads.
_DOCUMENT_JSON_DESC = (
    "Optional baseline ATRIUM Document JSON (accretion model, docs/document_schema.md / "
    "issue #13). When given, the response's `document_json` carries the record back with only "
    "page-classification's `page_categories` block and `pages[].category` / "
    "`pages[].category_confidence` fields updated — every other tool's block passes through "
    "untouched. A baseline that does not validate against atrium_document.schema.json is still "
    "accepted (rule 6), but the response then also carries `document_json_schema_error`."
)
_DOCUMENT_JSON_OUT_DESC = (
    "Return a document record even with no baseline uploaded. page-classification is stage 1 "
    "of the pipeline, so it ORIGINATES the record (the E2E smoke passes it "
    "`--document-json-out` alone); without this flag the service could accrete onto someone "
    "else's record but never start one. Mirrors the CLI's `--document-json-out`."
)


async def _document_json_part(
    document_json: Optional[UploadFile],
    document_json_out: bool,
    doc_id: str,
    pages: Sequence[Tuple[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Run this tool's blocks through the CLI's own accretion path, or do nothing.

    Opt-in, like translator's and llm-enrich's services: a caller that sends neither part gets
    the exact response shape it got before, so this is additive on the wire.
    """
    if document_json is None and not document_json_out:
        return None, None

    baseline_bytes = None
    if document_json is not None:
        # `or None`: some clients send the multipart field with an empty body rather than
        # omitting it. That means "no baseline", not "a baseline that is zero bytes long" —
        # taken literally it reaches load_document() and dies on a JSONDecodeError.
        baseline_bytes = await document_json.read() or None

    try:
        return build_document_record(doc_id, pages, baseline_bytes)
    except ValueError as exc:
        # Unparseable JSON (JSONDecodeError is a ValueError) or a schema_version newer than this
        # tool understands. Both are the CALLER's payload, so §4.4 says 422, not 500 — and
        # certainly not the endpoint's blanket "Error processing image.", which would send
        # somebody debugging their upload to look at the classifier.
        raise HTTPException(status_code=422, detail=f"Unusable document_json baseline: {exc}") from exc
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


@app.get("/info")
def get_info():
    """Return service identity, capabilities, and available model versions (§4.1)."""
    # [FIX]: Removed the hardcoded fallback list.
    # model_registry is the single source of truth.
    from model_registry import CATEGORIES

    model_info = {v: manager.get_model_details(v) for v in manager.available_versions}
    model_info["all"] = manager.get_model_details("all")

    return build_info(
        app,
        service="atrium-page-classification",
        limits={"max_upload_mb": MAX_UPLOAD_MB, "max_pdf_pages": MAX_PDF_PAGES},
        categories=CATEGORIES,
        available_models=model_info,
    )


@app.post("/predict_image", response_model=ImageResponse)
async def predict_image(
    version: str = Form("all"),
    topn: int = Form(3),
    file: UploadFile = File(...),
    document_json: UploadFile = File(None, description=_DOCUMENT_JSON_DESC),
    document_json_out: bool = Form(False, description=_DOCUMENT_JSON_OUT_DESC),
):
    """Classify a single uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB."
        )

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        predictions = manager.predict(image, version=version, topn=topn)
        if isinstance(predictions, dict) and "error" in predictions:
            raise HTTPException(status_code=500, detail=predictions["error"])

        # A single image is ONE PAGE of a document, and which page is carried in its filename
        # — so the doc_id/page split is utils.doc_id_and_page(), the same derivation run.py's
        # -f path uses, or the service would fork the record it is supposed to accrete onto.
        doc_id, page_key = doc_id_for_image(file.filename)
        record, schema_err = await _document_json_part(
            document_json, document_json_out, doc_id, [(page_key, predictions)]
        )

        return ImageResponse(
            type="image",
            predictions=predictions,
            document_json=record,
            document_json_schema_error=schema_err,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image.")


@app.post("/predict_document")
async def predict_document(
    version: str = Form("all"),
    topn: int = Form(3),
    file: UploadFile = File(...),
    document_json: UploadFile = File(None, description=_DOCUMENT_JSON_DESC),
    document_json_out: bool = Form(False, description=_DOCUMENT_JSON_OUT_DESC),
):
    """Extracts pages from a PDF and classifies each page."""
    if not file.content_type or file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB."
        )

    try:
        import fitz  # PyMuPDF

        pdf_document = fitz.open(stream=content, filetype="pdf")

        if len(pdf_document) > MAX_PDF_PAGES:
            raise HTTPException(status_code=413, detail=f"PDF has too many pages. Limit is {MAX_PDF_PAGES}.")

        page_results = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            predictions = manager.predict(img, version=version, topn=topn)
            page_results.append({"page": page_num + 1, "predictions": predictions})

        # A PDF is a whole document: the page numbers are its own 1..N, so no filename
        # page-split here — just the canonical doc_id.
        record, schema_err = await _document_json_part(
            document_json,
            document_json_out,
            doc_id_for_document(file.filename),
            [(str(r["page"]), r["predictions"]) for r in page_results],
        )

        response: Dict[str, Any] = {"type": "document", "pages": page_results}
        if record is not None:
            response["document_json"] = record
        if schema_err:
            response["document_json_schema_error"] = schema_err
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="Error processing document.")


if __name__ == "__main__":
    # service/README.md documents `python3 api.py` as step 1 of running the service, and this
    # file had no __main__ block at all — the command it tells the user to run exited silently
    # having started nothing (atrium-project#10, G3, same class as the absent uvicorn: a
    # documented start path that does not start anything).
    #
    # The app OBJECT, not the "service.api:app" import string: the README says to run this from
    # the service/ directory, where that string does not resolve (there is no `service` package
    # below service/). Passing the object works from either cwd and costs only --reload, which
    # a production entrypoint should not have anyway — docker-compose.yml's `api` service and
    # the setup script's closing instructions both use the uvicorn CLI for that.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
