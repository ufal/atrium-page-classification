# Atrium Page Classification API Service 🚀

### Goal: Serve historical document classification models via a lightweight REST API

**Scope:** This service provides a **FastAPI** interface for the Atrium Page Classification models. 
It allows users to upload document images and receive structural classification predictions (e.g., 
Text, Drawing, Table) using various fine-tuned deep learning models (ViT, EfficientNet, RegNetY). 
It includes a basic static HTML frontend for testing.

### Table of contents 📑

* [Service Description 📇](#service-description-)
* [Directory Structure 📂](#directory-structure-)
* [Installation & Setup 🛠](#installation--setup-)
* [Running the Server 🚀](#running-the-server-)
* [API Usage 📡](#api-usage-)
    * [Endpoints 🔗](#endpoints-)
    * [Request Example 💻](#request-example-)
* [Supported Models 🧠](#supported-models-)
* [Categories 🪧](#categories-)
* [Client Side Development 🎨](#client-side-development-)

---

## Service Description 📇

The API is built using **FastAPI** and is designed to run inference on single images. 
It acts as a bridge between the fine-tuned PyTorch models and downstream applications or web interfaces.

Key features:
* **Multiple Architectures:** Supports switching between ViT, RegNetY, and EfficientNet models dynamically.
* **GPU Support:** Automatically detects and utilizes CUDA devices if available.
* **Static Interface:** Serves a built-in HTML testing tool.


## Directory Structure 📂

The service is designed to sit within the larger `atrium-page-classification` structure. The API logic resides in the `service/` directory, while models are loaded from the parent `model/` directory.

```text
atrium-page-classification/
├── model/                   # 📦 Fine-tuned model weights (e.g., model_v53/)
├── service/                 # 🚀 API Source Code
│   ├── api.py               # FastAPI application entry point
│   ├── inference.py         # Model loading and prediction logic
│   ├── requirements.txt     # Python dependencies
│   └── static/              # 🎨 Frontend assets
│       ├── index.html       # Web interface
│       └── script.js        # Logic for the web interface
└── classifier.py            # Base ImageClassifier class (imported by inference.py)
```

## Installation & Setup 🛠

### 1. Prerequisites
* **Python 3.10+**
* **CUDA-capable GPU** (Recommended for inference speed, though CPU is supported).

### 2. Install Dependencies
Navigate to the root `atrium-page-classification` directory and install the required packages 
using the service requirements file:

```bash
pip install -r service/requirements.txt
```

Key libraries include: fastapi, uvicorn, python-multipart, pillow, torch, timm, transformers.

### 3. Model Weights

Ensure that the fine-tuned model weights are available in the `model/` directory.

Follow the instructions in the main repository to load the model weights correctly.

## Running the Server 🚀

To start the API server with hot-reloading enabled (useful for development):

```bash
uvicorn service.api:app --reload
```

The server will start at http://0.0.0.0:8000 (access to use the built-in visual testing tool).

## API Usage 📡

### Endpoints 🔗

| Method | Path       | Description                                                                                  |
|:-------|:-----------|:---------------------------------------------------------------------------------------------|
| `GET`  | `/`        | Serves the static `index.html` interface for manual testing.                                 |
| `GET`  | `/info`    | Returns metadata about available models and the active computation device (`cpu` or `cuda`). |
| `POST` | `/predict` | Performs inference on an uploaded image.                                                     |

### Request Example 💻

**Endpoint:** `/predict`

**Parameters (Form Data):**
* `file`: The image file (JPEG or PNG).
* `version`: The model version string (e.g., `v5.3`, `v1.3`) or `all`.
* `topn`: (Optional) Number of top predictions to return (Default: 3).

**Response (JSON):**

```json
{
    "filename": "sample_page.jpg",
    "requested_version": "v5.3",
    "predictions": {
        "v5.3": [
            { "label": "TEXT_T", "score": 0.985 },
            { "label": "TEXT", "score": 0.012 },
            { "label": "LINE_T", "score": 0.003 }
        ]
    }
}
```

## Supported Models 🧠

The API exposes specific model versions defined in `inference.py`. These map to different underlying 
base architectures, allowing users to balance speed vs. accuracy.

| Version  | Base Architecture                   | Description                                     |
|:---------|:------------------------------------|:------------------------------------------------|
| **v5.3** | `vit-large-patch16-384`             | Most accurate, slowest inference.               |
| **v4.3** | `regnety_160.swag_ft_in1k`          | Balanced option. Best performing "Small" model. |
| **v1.3** | `tf_efficientnetv2_m.in21k_ft_in1k` | CNN-based, faster inference.                    |
| **v2.3** | `vit-base-patch16-224`              | Standard Transformer baseline.                  |
| **v3.3** | `vit-base-patch16-384`              | Higher resolution Transformer baseline.         |

## Categories 🪧

The models classify pages into 11 distinct structural categories:

| Label     | Description                                                |
|:----------|:-----------------------------------------------------------|
| `TEXT`    | 📰 Mixed text (printed, handwritten, typed).               |
| `TEXT_T`  | 📄 Typed text (machine-typed paragraphs).                  |
| `TEXT_P`  | 📄 Printed text (published paragraphs).                    |
| `TEXT_HW` | ✏️📄 Handwritten text (paragraphs).                        |
| `LINE_T`  | 📏 Typed Table.                                            |
| `LINE_P`  | 📏 Printed Table.                                          |
| `LINE_HW` | ✏️📏 Handwritten Table.                                    |
| `DRAW`    | 📈 Drawing (maps, paintings, schematics).                  |
| `DRAW_L`  | 📈📏 Structured Drawing (drawings within a layout/legend). |
| `PHOTO`   | 🌄 Photo (photographs/cutouts).                            |
| `PHOTO_L` | 🌄📏 Structured Photo (photos in a table layout).          |


## Client Side Development 🎨

This API service includes a lightweight vanilla JS frontend (`service/static/script.js`) for immediate testing. 
However, the full LINDAT client integration is developed separately.

For client-side development instructions, please refer to the **LINDAT Common Development Guide**:
[https://github.com/ufal/lindat-common/?tab=readme-ov-file#development](https://github.com/ufal/lindat-common/?tab=readme-ov-file#development).
