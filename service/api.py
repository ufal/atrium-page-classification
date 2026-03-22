import os
import sys
import shutil
import tempfile
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes
import time

# FIX: guard relative import so the module works both when launched via
#   `uvicorn service.api:app`  (package mode, relative import works)
#   `python api.py`            (direct execution, needs absolute import)
try:
    from .inference import manager, AVAILABLE_VERSIONS, CATEGORIES
except ImportError:
    # Fallback for direct execution: add this file's directory to sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference import manager, AVAILABLE_VERSIONS, CATEGORIES

app = FastAPI(title="Atrium Page Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "frontend")
LINDAT_DIST_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../dist"))

CATEGORY_DESCRIPTIONS = {
    "TEXT": "📰 Mixed text (printed, handwritten, typed).",
    "TEXT_T": "📄 Typed text (machine-typed paragraphs).",
    "TEXT_P": "📄 Printed text (published paragraphs).",
    "TEXT_HW": "✏️📄 Handwritten text (paragraphs).",
    "LINE_T": "📏 Typed Table.",
    "LINE_P": "📏 Printed Table.",
    "LINE_HW": "✏️📏 Handwritten Table.",
    "DRAW": "📈 Drawing (maps, paintings, schematics).",
    "DRAW_L": "📈📏 Structured Drawing (drawings within a layout/legend).",
    "PHOTO": "🌄 Photo (photographs/cutouts).",
    "PHOTO_L": "🌄📏 Structured Photo (photos in a table layout)."
}

if os.path.exists(STATIC_DIR):
    app.mount("/frontend", StaticFiles(directory=STATIC_DIR), name="frontend")

if os.path.exists(LINDAT_DIST_DIR):
    app.mount("/dist", StaticFiles(directory=LINDAT_DIST_DIR), name="lindat-dist")


# --- REFINED: Use environment variables for CORS, fallback to local dev defaults ---
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Pre-load the default model at startup to avoid a latency spike on the
    first request.  Only v4.3 is warmed up by default; extend the list if
    multi-model latency on startup is acceptable."""
    manager.warmup(["v4.3"])


@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return {"message": "API Running. Index not found."}


@app.get("/info")
async def info():
    return {
        "available_models": AVAILABLE_VERSIONS,
        "device": manager.device,
        "categories": CATEGORY_DESCRIPTIONS
    }


@app.post("/predict_image")
async def predict_image(
        file: UploadFile = File(...),
        version: str = Form(...),
        topn: int = Form(3)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG/PNG allowed.")

    if topn < 1: topn = 1

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        results = manager.predict(image, version, topn)

        if isinstance(results, dict) and "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        for res in results:
            res["description"] = CATEGORY_DESCRIPTIONS.get(res["label"], "")

        return {
            "type": "image",
            "model_version": manager.get_model_details(version),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/predict_document")
async def predict_document(
        file: UploadFile = File(...),
        version: str = Form(...),
        topn: int = Form(3)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF allowed.")

    if topn < 1: topn = 1

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            contents = await file.read()

            # Convert PDF to images
            try:
                images = convert_from_bytes(contents)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to convert PDF (poppler installed?): {str(e)}")

            if not images:
                raise HTTPException(status_code=400, detail="PDF contains no readable pages.")

            # Save pages as images for batch processing
            for i, img in enumerate(images):
                img.save(os.path.join(temp_dir, f"page_{i:04d}.png"))

            # Run Batch Inference
            batch_predictions = manager.predict_directory(temp_dir, version, topn)

            formatted_pages = []
            for i, preds in enumerate(batch_predictions):
                # --- Generate Thumbnail ---
                original_img = images[i]

                # Resize for thumbnail (e.g., max height 300px to keep JSON light)
                thumb = original_img.copy()
                thumb.thumbnail((300, 300))

                # Encode to Base64
                buffered = BytesIO()
                thumb.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                # -------------------------

                formatted_pages.append({
                    "page": i + 1,
                    "predictions": preds,
                    "thumbnail": img_str
                })

            return {
                "type": "document",
                "filename": file.filename,
                "model_version": manager.get_model_details(version),
                "requested_topn": topn,
                "pages": formatted_pages
            }

        except Exception as e:
            # Log the full error for debugging
            print(f"Error processing PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)