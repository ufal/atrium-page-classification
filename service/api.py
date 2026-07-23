import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

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
async def predict_image(version: str = Form("all"), topn: int = Form(3), file: UploadFile = File(...)):
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

        return ImageResponse(type="image", predictions=predictions)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image.")


@app.post("/predict_document")
async def predict_document(version: str = Form("all"), topn: int = Form(3), file: UploadFile = File(...)):
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

        return {"type": "document", "pages": page_results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="Error processing document.")
