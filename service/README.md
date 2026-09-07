
# ATRIUM Page Classification API Service 🚀

### Goal: Serve historical document classification models via a lightweight REST API

**Scope:** This service provides a **FastAPI** interface for the Atrium Page Classification models.
It allows users to upload document images and receive structural classification predictions (e.g.,
Text, Drawing, Table) using various fine-tuned on historical data [^17] deep learning models
(ViT, EfficientNet, RegNetY). It includes basic static HTML frontends for both standalone testing and LINDAT integration.

### Table of contents 📑

- [Service Description 📇](#service-description-)
- [Directory Structure 📂](#directory-structure-)
- [Supported Models 🧠](#supported-models-)
- [Categories 🪧](#categories-)
- [API Usage 📡](#api-usage-)
- [Installation & Setup 🛠](#installation--setup-)
- [Quick API Test Launch 🚀](#quick-api-test-launch-)
- [Client Side Test 🎨](#client-side-test-)
- [Contacts 📧](#contacts-)
- [Acknowledgements 🙏](#acknowledgements-)

---

## Service Description 📇

The API is built using **FastAPI** and is designed to run inference on single images or multipage PDFs.
It acts as a bridge between the fine-tuned PyTorch models and downstream applications or web interfaces.

Key features:
* **Multiple Architectures:** Supports switching between ViT, RegNetY, and EfficientNet models dynamically.
* **GPU Support:** Automatically detects and utilizes CUDA devices if available.
* **Lightweight Frontends:** Includes simple HTML/JS interfaces for manual testing of the API, both as standalone and LINDAT-ready modules.

## Directory Structure 📂

The service is designed to sit within the larger `atrium-page-classification` structure. The API logic resides in the `service/` directory, while models are loaded from the parent `model/` directory.

```text
atrium-page-classification/
├── model/                   # 📦 Fine-tuned model weights (e.g., model_v53/)
├── service/                 # 🚀 API Source Code
│   ├── api.py               # FastAPI application entry point
│   ├── inference.py         # Model loading and prediction logic
│   ├── requirements.txt     # Python dependencies for the API
│   ├── api_client.py        # Client script to test the API endpoints
│   ├── frontend/            # 🎨 Standalone frontend assets (LINDAT-independent)
│   │   ├── index.html       # Standalone web interface
│   │   └── script.js        # Standalone logic
│   └── frontend-lindat/     # 🎨 LINDAT-integrated frontend assets
│       ├── index.html       # Web interface with LINDAT headers/footers
│       └── script.js        # Logic handling LINDAT stylings
├── setup/                   # ⚙️ Project configuration & setup scripts
│   └── setup_api_service.sh # Setup script for environment, dependencies, and models
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

| Method | Path                | Description                                                                                                                                                                 |
|:-------|:--------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `GET`  | `/`                 | Serves the static `index.html` interface for manual testing.                                                                                                                |
| `GET`  | `/info`             | Service identity + capabilities: `service`, `version`, `endpoints`, `limits`, plus available models and device.                                                             |
| `GET`  | `/health`           | Liveness probe — 200 always, even mid-shutdown. `?deep=true` also checks at least one model version is loaded (503 on failure or while draining).                           |
| `GET`  | `/ready`            | Readiness probe (issue #55) — 503 until model warmup finishes, 200 while serving, 503 the instant `SIGTERM` arrives. The Kubernetes `readinessProbe`/`startupProbe` target. |
| `POST` | `/predict_image`    | Performs inference on an uploaded single image (JPG/PNG).                                                                                                                   |
| `POST` | `/predict_document` | Performs inference on an uploaded multipage PDF document.                                                                                                                   |

### Request Example 💻

**Endpoint:** `/predict_image`

**Parameters (Form Data):**
* `file`: The image file (JPEG or PNG).
* `version`: The model version string (e.g., `v5.3`, `v1.3`) or `all`.
* `topn`: (Optional) Number of top predictions to return (Default: 3).
* `document_json`: (Optional) A baseline **ATRIUM Document JSON** record to accrete onto — the
  service equivalent of the CLI's `--document-json`.
* `document_json_out`: (Optional, boolean) Return a record even with no baseline uploaded — the
  equivalent of `--document-json-out`. page-classification is stage 1 of the pipeline, so it
  *originates* the record as often as it updates one.

Request example using `curl`:

```bash
curl -X POST "http://localhost:8000/predict_image" \
  -F "file=@/path/to/image.png" \
  -F "version=v2.3" \
  -F "topn=1"
```

Example JSON response:
```json
{
  "type": "image",
  "model_version": "google/vit-base-patch16-224 (v2.3)",
  "requested_topn": 1,
  "predictions": [
    {
      "label": "TEXT",
      "score": 0.975
    }
  ]
}
```

### ATRIUM Document JSON accretion 🧩

Both `POST` endpoints implement accretion-contract rule 1 — *"services accept and return an
optional `document_json` part"* — so a pipeline can thread one per-document record through the
API exactly as it does through the CLI. This was missing entirely until
[atrium-project#10](https://github.com/ufal/atrium-project/issues/10) (finding J2): the CLI
implemented the contract in full while the deployed API surface implemented none of it.

```bash
# accrete onto an upstream record (alto-postprocess → page-classification)
curl -X POST "http://localhost:8000/predict_image" \
  -F "file=@CTX000000001_0007.png" \
  -F "document_json=@CTX000000001.document.json" \
  -F "version=v4.3"

# or originate one (stage 1, no baseline to inherit)
curl -X POST "http://localhost:8000/predict_image" \
  -F "file=@CTX000000001_0007.png" \
  -F "document_json_out=true"
```

The response then carries the updated record under `document_json`:

* only this tool's own contributions are written — the whole `page_categories` block, plus
  `pages[].category` and `pages[].category_confidence` at field level. Every other tool's block
  (`lines`, `entities`, `translations`, …) passes through byte-for-byte;
* the record's `doc_id` is derived with the same composition the CLI uses
  (`utils.doc_id_and_page`, which strips the page label and then defers to the shared
  `canonical_doc_id()`), so a service upload and a CLI run over the same file update the *same*
  record instead of forking it;
* a record page-classification builds that does not validate against
  `atrium_document.schema.json` is never returned — the request fails with `500` instead
  (Layer D). A **baseline** that does not validate is still accepted (rule 6: pass unknown
  content through), and the response then also carries `document_json_schema_error` naming the
  problem, so an automated caller can notice without reading the service log.


## Installation & Setup 🛠

### 1. Prerequisites
* **Python 3.10+**
* **NodeJS** (For client-side development within LINDAT environment)
* **Standard CPU** (Sufficient for **Client-side** development).
* **CUDA-capable GPU** (Recommended for **Server-side** inference speed, though CPU is supported). [^10]

### 2. Install Server Dependencies

Navigate to the root `atrium-page-classification` directory, then run a setup script to
create a virtual environment, and install all of the required packages:

```bash
# Create and activate virtual environment
git clone [https://github.com/ufal/atrium-page-classification.git](https://github.com/ufal/atrium-page-classification.git)
cd atrium-page-classification
chmod +x ./setup/setup_api_service.sh
./setup/setup_api_service.sh
```

Key libraries include: fastapi, uvicorn, python-multipart, pillow, PyMuPDF, torch, timm,
transformers. The serving half is in `service/requirements.txt` and the model stack in
`setup/requirements.txt`; the setup script installs both, and so does the Docker image.

> [!NOTE] `service/requirements.txt` had been pruned down to six pytest/contract packages —
> **no `uvicorn` at all** — while this page and `docker-compose.yml` both still told you to run
> it, so `docker compose --profile api up api` failed at container start
> ([atrium-project#10](https://github.com/ufal/atrium-project/issues/10), finding G3). The
> runtime set is restored, the contract test deps live in `setup/requirements-test.txt`, and
> `tests/test_service_runtime_deps.py` now asserts that every entrypoint the compose files and
> setup script invoke is actually declared somewhere the image installs from.

> [!NOTE] The virtual environment name is stated in the setup script and can be changed to an existing
> one if needed.

### 3. Model Weights

The setup script also downloads the fine-tuned model weights from the Hugging Face Hub [^1].
It is done via the `run.py` script that saves the weights in the `model/` directory.

> [!NOTE] The very first run may take some time as it downloads multiple model files to
> be cached locally. When using the WEB UI, `inference.py` will check for the models
> in the `model/` directory, and if not found, it will attempt to download them from
> Hugging Face Hub automatically.

If you prefer the manual approach, you can download the weights to the `model/` directory by yourself:

```bash
source venv/bin/activate
python3 run.py --hf -rev vX.3
````
where `X` is the model version (1, 2, 3, 4, or 5).

## Quick API Test Launch 🚀

Use this guide to verify the inference service is communicating correctly with the model manager.

### Launch Instructions

Open two terminal windows (or tabs) and run the following commands:

```bash
source venv/bin/activate
cd atrium-page-classification/service/
```

Then, in each window, execute the respective commands:


| **Server Console (Window 1)**                                                                                                                                                                         | **Client Console (Window 2)**                                                                                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Start the API:**<br><br>Run the FastAPI server from the service directory.<br><br>`python3 api.py`<br><br>You should see startup logs indicating the server is running on `http://0.0.0.0:8000`. | **2. Send a Request:**<br><br> Top-3 Classification of `image.png`:<br><br>`python3 api_client.py -f .../image.png -v v5.3 --top 3`<br><br> where `-f` and `-v` stand for **input file** and **model version**. |

### Expected Output

```json
{
  "type": "image",
  "model_version": "google/vit-large-patch16-384 (v5.3)",
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
  "type": "image",
  "model_version": "Ensemble (Average of 5 Models)",
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

## Shutdown behavior 🛑

Issue [#55](https://github.com/ufal/atrium-project/issues/55). The published `api` image
(`ghcr.io/ufal/atrium-page-classification:<version>-api`, new in that issue — before it
this service was only reachable via a compose entrypoint override, so no API image
existed to deploy) declares `HEALTHCHECK` (shallow `GET /health`, via the vendored
`service/healthcheck.py`) and `STOPSIGNAL SIGTERM`, and its `ENTRYPOINT` passes
`--timeout-graceful-shutdown 20`.

On `SIGTERM` the service flips `GET /ready` to **503** at once so an orchestrator stops
routing to it, answers new `/predict_*` calls with 503 ("retry against a live replica"),
and lets uvicorn finish in-flight classification before exiting. `GET /health`
deliberately stays 200 throughout — a liveness probe failing mid-shutdown would get the
container killed before the drain completed.

Inference now runs in a worker thread (`asyncio.to_thread`) rather than inline on the
event loop. That was a prerequisite, not a tidy-up: uvicorn's `SIGTERM` handler is an
event-loop callback, so while a synchronous `manager.predict()` held the loop, the signal
could not be processed at all and `--timeout-graceful-shutdown` had nothing to measure.

⚠️ A `/predict_document` call classifies up to `MAX_PDF_PAGES` (50) pages sequentially and
can legitimately outlive the 20s drain budget. Raise `--timeout-graceful-shutdown` and the
deployment's grace period together for that workload — see `docs/k8s_deployment.md` in the
hub.

A clean shutdown exits **143** (128 + SIGTERM), not 0: uvicorn re-raises the captured
signal on purpose so a supervisor sees the real cause. That is a normal stop, not a crash.

## Client Side Test 🎨

This API service includes two versions of the frontend for immediate testing:
1. `service/frontend/`: A lightweight, standalone vanilla JS frontend.
2. `service/frontend-lindat/`: A LINDAT-integrated client developed for usage inside the LINDAT ecosystem [^5].

For client-side development within LINDAT, open a **second console window** and follow these steps:

1.  **Clone the repository** and place `atrium-page-classification` project files into the `lindat-common` directory:
    ```bash
    git clone [https://github.com/ufal/lindat-common.git](https://github.com/ufal/lindat-common.git)
    cd lindat-common
    cp -r ../atrium-page-classification .
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
source venv/bin/activate
uvicorn service.api:app --reload
```

The server will start at `http://0.0.0.0:8000` (access this to use the built-in standalone visual testing tool located in `service/frontend`).

### Using the LINDAT client-side test interface

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

Follow the file tree to the `atrium-page-classification/service/frontend-lindat` directory. The frontend interface will open and allow you to upload images and test the API.

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
