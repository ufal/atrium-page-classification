import configparser
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# [FIX]: Use a relative import to support pytest running from the repo root,
# with a fallback for direct script execution.
try:
    from .inference import manager
except ImportError:
    from inference import manager

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PDF_PAGES = 50

SERVICE_NAME = "atrium-page-classification"
API_ENDPOINTS = ["/info", "/health", "/predict_image", "/predict_document"]


def _read_tool_version() -> str:
    """Read the tool version from setup/para_config.txt [tool] section.

    Single source of truth — security.reusable.yml already validates this value
    against CITATION.cff and the release tag, so the API version can never drift
    from the released version again.
    """
    config = configparser.ConfigParser()
    config.read(
        Path(__file__).resolve().parent.parent / "setup" / "para_config.txt",
        encoding="utf-8",
    )
    version = config.get("tool", "version", fallback="unknown")
    return version[1:] if version.lower().startswith("v") else version


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
    version=_read_tool_version(),
    description="API for classifying historical document page images.",
    lifespan=lifespan,
)

# CORS hardening
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
# A wildcard origin must not be combined with credentials (browsers reject it).
if (
    "*" in ALLOWED_ORIGINS
    and os.environ.get("ALLOW_CREDENTIALS", "true").lower() == "true"
):
    ALLOWED_ORIGINS.remove("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount frontend
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount(
        "/frontend",
        StaticFiles(directory=str(frontend_dir), html=True),
        name="frontend",
    )


class PredictionResult(BaseModel):
    label: str
    score: float


class ImageResponse(BaseModel):
    type: str
    predictions: List[PredictionResult]


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the ATRIUM Page Classification API. Use /info for available models."
    }


@app.get("/info")
def get_info():
    """Return service identity, capabilities, and limits (ATRIUM meta-contract)."""
    # [FIX]: Removed the hardcoded fallback list.
    # model_registry is the single source of truth.
    from model_registry import CATEGORIES

    model_info = {v: manager.get_model_details(v) for v in manager.available_versions}
    model_info["all"] = manager.get_model_details("all")

    return {
        "service": SERVICE_NAME,
        "version": app.version,
        "endpoints": API_ENDPOINTS,
        "limits": {"max_upload_mb": MAX_UPLOAD_MB, "max_pdf_pages": MAX_PDF_PAGES},
        "categories": CATEGORIES,
        "available_models": model_info,
    }


@app.get("/health")
def get_health(deep: bool = False):
    """Liveness (shallow) / readiness (deep=true, models loaded) probe."""
    if not deep:
        return {"status": "ok"}

    loaded = sorted(manager.models.keys())
    if not loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "detail": "no model loaded yet (warmup pending or failed)",
                "device": manager.device,
            },
        )
    return {
        "status": "ok",
        "models_loaded": loaded,
        "models_available": manager.available_versions,
        "device": manager.device,
    }


@app.post("/predict_image", response_model=ImageResponse)
async def predict_image(
    version: str = Form("all"), topn: int = Form(3), file: UploadFile = File(...)
):
    """Classify a single uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type. Please upload a PNG or JPEG image.",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
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
async def predict_document(
    version: str = Form("all"), topn: int = Form(3), file: UploadFile = File(...)
):
    """Extracts pages from a PDF and classifies each page."""
    if not file.content_type or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=415, detail="Unsupported media type. Please upload a PDF."
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )

    try:
        import fitz  # PyMuPDF

        pdf_document = fitz.open(stream=content, filetype="pdf")

        if len(pdf_document) > MAX_PDF_PAGES:
            raise HTTPException(
                status_code=413,
                detail=f"PDF has too many pages. Limit is {MAX_PDF_PAGES}.",
            )

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
