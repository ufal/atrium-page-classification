import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from .inference import manager, AVAILABLE_VERSIONS

app = FastAPI(title="Atrium Page Classification API")

# Allow CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
LINDAT_DIST_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../dist"))

# --- MOUNTS ---
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if os.path.exists(LINDAT_DIST_DIR):
    app.mount("/dist", StaticFiles(directory=LINDAT_DIST_DIR), name="lindat-dist")

# --- ROUTES ---

@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            content=f"""
            <h1>API Running</h1>
            <p>Static files not found at {STATIC_DIR}. Please ensure index.html exists.</p>
            <p>Available versions: {AVAILABLE_VERSIONS}</p>
            """, 
            status_code=200
        )
        
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/info")
async def info():
    return {
        "available_models": AVAILABLE_VERSIONS,
        "device": manager.device
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    version: str = Form(...),
    topn: int = Form(3)
):
    # Validate File Type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG/PNG allowed.")

    # Validate Version
    valid_versions = AVAILABLE_VERSIONS + ["all"]
    if version not in valid_versions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid version '{version}'. Choose from {valid_versions}"
        )

    # Process Image
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Corrupt image file: {str(e)}")

    # Run Inference
    results = manager.predict(image, version, topn)
    
    return {
        "filename": file.filename,
        "requested_version": version,
        "predictions": results
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server. Models will be loaded from: {os.path.abspath(os.path.join(BASE_DIR, '../model'))}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
