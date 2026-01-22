# Atrium Page Classification API Service 🚀

### Goal: Serve historical document classification models via a lightweight REST API

**Scope:** This service provides a **FastAPI** interface for the Atrium Page Classification models. 
It allows users to upload document images and receive structural classification predictions (e.g., 
Text, Drawing, Table) using various fine-tuned on historical data [^17] deep learning models 
(ViT, EfficientNet, RegNetY). It includes a basic static HTML frontend for testing.

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
* **CUDA-capable GPU** (Recommended for inference speed, though CPU is supported). [^10]
* **Python 3.10+**
* **NodeJS** (For client-side development)
* **CUDA-capable GPU** (Recommended for **Server-side** inference speed, though CPU is supported).
* **Standard CPU** (Sufficient for **Client-side** development).

### 2. Install Server Dependencies

Navigate to the root `atrium-page-classification` directory, create a virtual environment, 
and install the required packages:

```bash
# Create and activate virtual environment
cd atrium-page-classification
python -m venv venv-api
source venv-api/bin/activate
pip install -r service/requirements.txt
```

Key libraries include: fastapi, uvicorn, python-multipart, pillow, torch, timm, transformers.

### 3. Model Weights

Ensure that the fine-tuned model weights are available in the `model/` directory.

Follow the instructions in the main repository to load the model weights correctly.

## Running the Server 🚀

To start the API server with hot-reloading enabled (useful for development), ensure your virtual 
environment is activated in your **first console window**: [^3]

```bash
source venv-api/bin/activate
uvicorn service.api:app --reload
```

The server will start at http://0.0.0.0:8000 (access to use the built-in visual testing tool).

## Client Side Test 🎨

This API service includes a lightweight vanilla JS frontend (`service/static/script.js`) for immediate testing. 
However, the full LINDAT client integration is developed separately. [^5]

For client-side development, open a **second console window** and follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ufal/lindat-common.git](https://github.com/ufal/lindat-common.git)
    cd lindat-common
    ```

2.  **Install NodeJS environment** (unless you already have one):
    ```bash
    curl -o- [https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh](https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh) | bash
    nvm install stable
    nvm use stable
    ```

3.  **Install dependencies for development:**
    ```bash
    npm install
    ```

4. Move project files to `lindat-common` directory:
    ```bash
    cp -r atrium-page-classification .
   # or
   mv atrium-page-classification .
   ```

5. **Run development server:**
    ```bash
    make run
    ```

For further details, please refer to the **LINDAT Common Development Guide**:
[https://github.com/ufal/lindat-common/?tab=readme-ov-file#development](https://github.com/ufal/lindat-common/?tab=readme-ov-file#development).

### Using the client-side test interface

Assuming your **second console** output end like this:

```commandline
> lindat-common@3.5.0 start
> webpack-dev-server -p --debug --quiet

(node:2985155) Warning: `--localstorage-file` was provided without a valid path
(Use `node --trace-warnings ...` to show where the warning was created)
ℹ ｢wds｣: Project is running at http://localhost:8080/
ℹ ｢wds｣: webpack output is served from /
ℹ ｢wds｣: Content not from webpack is served from /home.../lindat-common

```

Open the URL `http://localhost:8080` in your web browser to access the LINDAT client interface. 

Follow the file tree to the `atrium-page-classification/service/static` directory. The frontend interface
will open and allow you to upload images and test the API.

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

## Contacts 📧

**For support write to:** lutsai.k@gmail.com responsible for this GitHub repository [^8] 🔗

## Acknowledgements 🙏

- **Developed by** UFAL [^7] 👥
- **Funded by** ATRIUM [^4]  💰
- **Shared by** ATRIUM [^4] & UFAL [^7] 🔗
- **Model type:** 
  - fine-tuned ViT with a 224x224 [^2] 🔗 or 384x384 [^13] [^14] 🔗 resolution size 
  - fine-tuned RegNetY-16GF with a 224x224 resolution [^18] or EffNetV2 with a 384x384 [^19] 🔗 resolution size 

**©️ 2026 UFAL & ATRIUM**

----

[^1]: https://huggingface.co/ufal/vit-historical-page
[^2]: https://huggingface.co/google/vit-base-patch16-224
[^3]: https://docs.python.org/3/library/venv.html
[^4]: https://atrium-research.eu/
[^5]: https://github.com/ufal/lindat-common
[^6]: https://www.ghostscript.com/releases/gsdnld.html
[^7]: https://ufal.mff.cuni.cz/home-page
[^8]: https://github.com/ufal/atrium-page-classification
[^10]: https://developer.nvidia.com/cuda-python
[^13]: https://huggingface.co/google/vit-base-patch16-384
[^14]: https://huggingface.co/google/vit-large-patch16-384
[^17]: http://hdl.handle.net/20.500.12800/1-5959
[^18]: https://huggingface.co/timm/regnety_160.swag_ft_in1k
[^19]: https://huggingface.co/timm/tf_efficientnetv2_m.in21k_ft_in1k

