# ATRIUM Page Classification API Service 🚀

### Goal: Serve historical document classification models via a lightweight REST API

**Scope:** This service provides a **FastAPI** interface for the Atrium Page Classification models. 
It allows users to upload document images and receive structural classification predictions (e.g., 
Text, Drawing, Table) using various fine-tuned on historical data [^17] deep learning models 
(ViT, EfficientNet, RegNetY). It includes a basic static HTML frontend for testing.

### Table of contents 📑

* [Service Description 📇](#service-description-)
* [Directory Structure 📂](#directory-structure-)
* [Supported Models 🧠](#supported-models-)
* [Categories 🪧](#categories-)
* [API Usage 📡](#api-usage-)
* [Installation & Setup 🛠](#installation--setup-)
* [Quick API Test Launch 🚀](#quick-api-test-launch-)
* [Client Side Test 🎨](#client-side-test-)
* [Contacts 📧](#contacts-)
* [Acknowledgements 🙏](#acknowledgements-)

---

## Service Description 📇

The API is built using **FastAPI** and is designed to run inference on single images. 
It acts as a bridge between the fine-tuned PyTorch models and downstream applications or web interfaces.

Key features:
* **Multiple Architectures:** Supports switching between ViT, RegNetY, and EfficientNet models dynamically.
* **GPU Support:** Automatically detects and utilizes CUDA devices if available.
* **Lightweight Frontend:** Includes a simple HTML/JS interface for manual testing of the API.

## Directory Structure 📂

The service is designed to sit within the larger `atrium-page-classification` structure. The API logic resides in the `service/` directory, while models are loaded from the parent `model/` directory.

```text
atrium-page-classification/
├── model/                   # 📦 Fine-tuned model weights (e.g., model_v53/)
├── service/                 # 🚀 API Source Code
│   ├── api.py               # FastAPI application entry point
│   ├── inference.py         # Model loading and prediction logic
│   ├── requirements.txt     # Python dependencies for the API
│   ├── test_api.py          # Client script to test the API endpoints
│   └── frontend/            # 🎨 Frontend assets
│       ├── index.html       # Web interface
│       └── script.js        # Logic for the web interface
├── setup_api_server.sh      # Setup script for environment, dependencies, and models
├── run.py                   # Script to download models manually
└── classifier.py            # Base ImageClassifier class (imported by inference.py)
```

## Supported Models 🧠

The API exposes specific model versions defined in `inference.py`. These map to different underlying 
base architectures, allowing users to balance speed vs. accuracy.

| Version  | Base Architecture                   | Description                                     |
|:---------|:------------------------------------|:------------------------------------------------|
| **v4.3** | `regnety_160.swag_ft_in1k`          | Balanced option. Best performing "Small" model. |
| **v2.3** | `vit-base-patch16-224`              | Standard Transformer baseline.                  |
| **v3.3** | `vit-base-patch16-384`              | Higher resolution Transformer baseline.         |
| **v5.3** | `vit-large-patch16-384`             | Most accurate, slowest inference.               |
| **v1.3** | `tf_efficientnetv2_m.in21k_ft_in1k` | CNN-based, faster inference.                    |

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

Request example using `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.png" \
  -F "version=v2.3" \
  -F "topn=1"
```

Example JSON response:
```json
{
  "model_version": "google/vit-base-patch16-224 (v2.3)",
  "best_category": "TEXT",
  "score": 0.975,
  "requested_topn": 1,
  "predictions": [
    {
      "label": "TEXT",
      "score": 0.975
    }
  ]
}
```

## Installation & Setup 🛠

### 1. Prerequisites
* **Python 3.10+**
* **NodeJS** (For client-side development)
* **Standard CPU** (Sufficient for **Client-side** development).
* **CUDA-capable GPU** (Recommended for **Server-side** inference speed, though CPU is supported). [^10]

### 2. Install Server Dependencies

Navigate to the root `atrium-page-classification` directory, then run a setup script to 
create a virtual environment, and install all of the required packages:

```bash
# Create and activate virtual environment
git clone https://github.com/ufal/atrium-page-classification.git
cd atrium-page-classification
chmod + x setup_api_server.sh
./setup_api_server.sh
```

Key libraries include: fastapi, uvicorn, python-multipart, pillow, torch, timm, transformers. These
libraries can be found in `service/requirements.txt` available for manual installation if needed.

> [!NOTE] The virtual environment name is stated in the setup script and can be changed to the already existing
> one if needed.

### 3. Model Weights

The setup script also downloads the fine-tuned model weights from the Hugging Face Hub [^1].
It is done via 'run.py' script that saves the weights in the `model/` directory.

> [!NOTE] The very first run may take some time as it downloads multiple model files to 
> be cached locally. When using the WEB UI, `inference.py` will check for the models 
> in the `model/` directory, and if not found, it will attempt to download them from
> Hugging Face Hub automatically.

If you prefer the manual approach, you can download the weights to the `model/` directory by yourself:

```bash
source venv-api/bin/activate
python3 run.py --hf -rev vX.3
````
where `X` is the model version (1, 2, 3, 4, or 5).

## Quick API Test Launch 🚀

Use this guide to verify the inference service is communicating correctly with the model manager.

### Launch Instructions

Open two terminal windows (or tabs) and run the following commands:

```bash
source venv-api/bin/activate
cd atrium-page-classification/service/
```

Then, in each window, execute the respective commands:


| **Server Console (Window 1)**                                                                                                                                                                         | **Client Console (Window 2)**                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Start the API:**<br><br>Run the FastAPI server from the service directory.<br><br>`python3 api.py`<br><br>You should see startup logs indicating the server is running on `http://0.0.0.0:8000`. | **2. Send a Request:**<br><br> Top-3 Classification of `image.png`:<br><br>`python3 test_api.py -<br/> -f .../image.png -v v5.3 --top 3`<br><br> where `-f` and `-v` stand for **input file** and **model version**. |

### Expected Output

```json
{
  "model_version": "google/vit-large-patch16-384 (v5.3)",
  "best_category": "TEXT",
  "score": 0.985,
  "requested_topn": 3,
  "predictions": [
    {
      "label": "TEXT",
      "score": 0.985
    },
    {
      "label": "TEXT_P",
      "score": 0.010
    },
    {
      "label": "LINE_P",
      "score": 0.002
    }
  ]
}
```
Or for `-v all` the best models ensemble (average of 5 class scores):

```json
{
  "model_version": "Ensemble (Average of 5 Models)",
  "best_category": "LINE_HW",
  "score": 0.9997688055038452,
  "requested_topn": 5,
  "predictions": [
    {"label": "LINE_HW",
      "score": 0.9997688055038452
    },
    {"label": "LINE_T",
      "score": 6.4081506344138e-05
    },
    {"label": "TEXT",
      "score": 4.853471283947641e-05
    },
    {"label": "DRAW_L",
      "score": 3.790065846882573e-05
    },
    {"label": "DRAW",
      "score": 3.168964675463182e-05
    }
  ]
}
```

## Client Side Test 🎨

This API service includes a lightweight vanilla JS frontend (`service/frontend/script.js`) for immediate testing. 
However, the full LINDAT client integration is developed separately. [^5]

For client-side development, open a **second console window** and follow these steps:

1.  **Clone the repository** and place `atrium-page-classification` project files to `lindat-common` directory
    ```bash
    git clone [https://github.com/ufal/lindat-common.git](https://github.com/ufal/lindat-common.git)
    cd lindat-common
    cp -r atrium-page-classification .
    # or
    mv atrium-page-classification .
    ```

2.  **Install NodeJS environment** (unless you already have one) and **Install dependencies for development:**
    ```bash
    curl -o- [https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh](https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh) | bash
    nvm install stable
    nvm use stable
    npm install
    ```

3. **Run development server:**
    ```bash
    make run
    ```

For further details, please refer to the **LINDAT Common Development Guide**:
[https://github.com/ufal/lindat-common/?tab=readme-ov-file#development](https://github.com/ufal/lindat-common/?tab=readme-ov-file#development).

### Running the Server 🚀

To start the API server with hot-reloading enabled (useful for development), ensure your virtual 
environment is activated in your **first console window**: [^3]

```bash
cd atrium-page-classification
source venv-api/bin/activate
uvicorn service.api:app --reload
```

The server will start at http://0.0.0.0:8000 (access to use the built-in visual testing tool).

### Using the client-side test interface

Assuming your **second console** output ends like this:

```commandline
> lindat-common@3.5.0 start
> webpack-dev-server -p --debug --quiet

(node:2985155) Warning: `--localstorage-file` was provided without a valid path
(Use `node --trace-warnings ...` to show where the warning was created)
> Project is running at http://localhost:8080/
> webpack output is served from /
> Content not from webpack is served from /home.../lindat-common
```

Open the URL `http://localhost:8080` in your web browser to access the LINDAT client interface. 

Follow the file tree to the `atrium-page-classification/service/frontend` directory. The frontend interface
will open and allow you to upload images and test the API.

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

