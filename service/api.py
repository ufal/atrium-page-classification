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

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB limit for safety
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
    version="1.4.0-beta",
    description="API for classifying historical document page images.",
    lifespan=lifespan
)

# CORS hardening
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
# A wildcard origin must not be combined with credentials (browsers reject it).
if "*" in ALLOWED_ORIGINS and os.environ.get("ALLOW_CREDENTIALS", "true").lower() == "true":
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
    """Return available model versions and categories."""
    try:
        from model_registry import CATEGORIES
    except ImportError:
        # Fallback if accessed abnormally
        CATEGORIES = ["DRAW", "DRAW_L", "LINE_HW", "LINE_P", "LINE_T", "PHOTO", "PHOTO_L", "TEXT", "TEXT_HW", "TEXT_P",
                      "TEXT_T"]

    model_info = {
        v: manager.get_model_details(v) for v in manager.available_versions
    }
    model_info["all"] = manager.get_model_details("all")

    return {
        "categories": CATEGORIES,
        "available_models": model_info
    }


@app.post("/predict_image", response_model=ImageResponse)
async def predict_image(
        version: str = Form("all"),
        topn: int = Form(3),
        file: UploadFile = File(...)
):
    """Classify a single uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.")

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
        version: str = Form("all"),
        topn: int = Form(3),
        file: UploadFile = File(...)
):
    """Extracts pages from a PDF and classifies each page."""
    if not file.content_type or file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.")

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
            page_results.append({
                "page": page_num + 1,
                "predictions": predictions
            })

        return {"type": "document", "pages": page_results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="Error processing document.")
