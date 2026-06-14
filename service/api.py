import os
import sys
import base64
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes

# FIX: guard relative import so the module works both when launched via
#   `uvicorn service.api:app`  (package mode, relative import works)
#   `python api.py`            (direct execution, needs absolute import)
try:
    from .inference import manager, AVAILABLE_VERSIONS, CATEGORIES
except ImportError:
    # Fallback for direct execution: add this file's directory to sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference import manager, AVAILABLE_VERSIONS, CATEGORIES

# ── REVIEW FIX (Major F/K): request limits to bound the DoS surface ──────────
# convert_from_bytes() rasterises every page of an uploaded PDF and we hold a
# base64 thumbnail per page in memory, so both the upload size and the page
# count must be capped.  Overridable via environment for larger deployments.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))   # 50 MB
MAX_PDF_PAGES    = int(os.getenv("MAX_PDF_PAGES", "200"))


# ── REVIEW FIX (Minor H/I): lifespan handler replaces the deprecated
#    @app.on_event("startup") hook.  Default model is warmed up on startup. ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.warmup(["v4.3"])
    yield


app = FastAPI(title="Atrium Page Classification API", lifespan=lifespan)

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


# ── REVIEW FIX (Major F/K): CORS is registered EXACTLY ONCE, from the
#    environment-driven allow-list.  The previous code added the middleware
#    twice — the first with allow_origins=["*"] — which silently defeated the
#    restricted second policy.  Default falls back to local dev origins. ──────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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

    if topn < 1:
        topn = 1

    try:
        contents = await file.read()

        # REVIEW FIX (Major F/K): enforce the upload-size cap.
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(contents)} bytes). "
                       f"Limit is {MAX_UPLOAD_BYTES} bytes.",
            )

        image = Image.open(BytesIO(contents)).convert("RGB")

        results = manager.predict(image, version, topn)

        if isinstance(results, dict) and "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        for res in results:
            res["description"] = CATEGORY_DESCRIPTIONS.get(res["label"], "")

        return {
            "type": "image",
            "model_version": manager.get_model_details(version),
            # REVIEW FIX (Minor F/J): echo requested_topn for parity with the
            # documented schema and the /predict_document response.
            "requested_topn": topn,
            "predictions": results
        }
    # REVIEW FIX (Major F): re-raise HTTPException unchanged.  The previous
    # broad `except Exception` caught our deliberate 500/413/400 and re-wrapped
    # them as 400, corrupting every status code.
    except HTTPException:
        raise
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

    if topn < 1:
        topn = 1

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            contents = await file.read()

            # REVIEW FIX (Major F/K): enforce the upload-size cap before the
            # (memory-heavy) rasterisation.
            if len(contents) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large ({len(contents)} bytes). "
                           f"Limit is {MAX_UPLOAD_BYTES} bytes.",
                )

            # Convert PDF to images
            try:
                images = convert_from_bytes(contents)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to convert PDF (poppler installed?): {str(e)}")

            if not images:
                raise HTTPException(status_code=400, detail="PDF contains no readable pages.")

            # REVIEW FIX (Major F/K): cap the page count to bound memory.
            if len(images) > MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDF has too many pages ({len(images)}). "
                           f"Limit is {MAX_PDF_PAGES}.",
                )

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

        # REVIEW FIX (Major F): re-raise HTTPException unchanged so the inner
        # 400/413 are not rewritten as a generic 500 by the broad handler below.
        except HTTPException:
            raise
        except Exception as e:
            # Log the full error for debugging
            print(f"Error processing PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)