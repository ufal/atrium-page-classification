---
name: atrium-page-classification
description: Classifies historical document page images (PNG/JPEG) and multipage PDFs into 11 structural categories (text, handwritten, tables, drawings, photos) using fine-tuned ViT / RegNetY / EffNetV2 models. Use this skill to route archival pages to the correct downstream processing pipeline (OCR, HTR, table extraction, image handling).
---

# ATRIUM Page Classification Skill 🪧

This skill provides agent access to the **ATRIUM Page Classification** service - fine-tuned
vision models for sorting scans of historical archive pages by their structural content.
It follows a **server-client** design: a FastAPI server (in `service/`) performs the heavy
inference, and a zero-dependency client script (`scripts/atrium_classify.py`) is the only
thing the agent calls directly.

## Operational Requirements ⚙️

- **Server**: A running instance of the API is required. By default the skill targets
  `http://localhost:8000`; override with `--base-url` or the `ATRIUM_PC_URL` environment
  variable (e.g. once a hosted LINDAT deployment is available).
- **Client dependencies**: None - `scripts/atrium_classify.py` uses only the Python 3
  standard library.
- **Server dependencies**: Docker (recommended) or a Python venv with
  `service/requirements.txt` (PyTorch, transformers, timm). GPU is optional; CPU works.
- **First launch**: Model weights are downloaded from Hugging Face
  (`ufal/vit-historical-page`) - roughly 0.2-1.2 GB per revision, 5 revisions for the
  ensemble. Warmup can take several minutes. Do **not** treat a slow first start as failure.
- **Upload limits**: 10 MB per file, 50 pages per PDF (server-enforced).

## Categories 🪧

|     Label | Meaning                                                              |
|----------:|:---------------------------------------------------------------------|
|    `DRAW` | 📈 drawings, maps, paintings, schematics (may contain text labels)   |
|  `DRAW_L` | 📈📏 drawings within a table-like layout or with a tabular legend    |
| `LINE_HW` | ✏️📏 handwritten text in tabular / form-like structure               |
|  `LINE_P` | 📏 printed text in tabular / form-like structure                     |
|  `LINE_T` | 📏 machine-typed text in tabular / form-like structure               |
|   `PHOTO` | 🌄 photographs or photographic cutouts (may have captions)           |
| `PHOTO_L` | 🌄📏 photos within a table-like layout or with tabular annotations   |
|    `TEXT` | 📰 mixed printed / handwritten / typed text, minor graphics possible |
| `TEXT_HW` | ✏️📄 only handwritten text in paragraph or block form                |
|  `TEXT_P` | 📄 only printed text in paragraph or block form                      |
|  `TEXT_T` | 📄 only machine-typed text in paragraph or block form                |

The distinction encodes three routing criteria: graphical elements (drawing vs. photo),
text type (handwritten / printed / typed / mixed), and tabular layout presence. Downstream
pipelines differ per category (e.g. HTR for `*_HW`, OCR for `TEXT_P`/`TEXT_T`, table
extraction for `LINE_*`).

## Workflows 🪄

### 1. Ensure the server is running

```bash
bash scripts/server.sh          # Docker CPU (or local uvicorn fallback)
bash scripts/server.sh --gpu    # Docker with GPU
bash scripts/server.sh --local  # force local uvicorn (no Docker)
```

The script is idempotent: if `GET /info` already answers, it exits immediately. It waits
up to 15 minutes for first-run warmup.

### 2. Classify

```bash
# Single image, best-5 ensemble, top-3 (defaults)
python3 scripts/atrium_classify.py page.png

# Multipage PDF, top-5, machine-readable CSV
python3 scripts/atrium_classify.py document.pdf --topn 5 --format csv

# Several images with a single specific model, raw JSON
python3 scripts/atrium_classify.py scans/*.png --version v4.3 --format json

# Discover available model versions and categories
python3 scripts/atrium_classify.py --info

# Remote server instead of localhost
python3 scripts/atrium_classify.py page.png --base-url https://example.org/atrium-pc
```

Output rows are `FILE, PAGE, RANK, LABEL, SCORE` (page is `1` for single images).

## Agent Guidelines 🤖

1. **Model selection**: Prefer the default `--version all` (average of the 5 best
   fine-tuned models) - it is the most accurate and calibrated option. Use a single
   version (e.g. `v4.3`) only when the user asks for a specific model, or for faster
   throughput on large batches.
2. **Top-N discipline**: Top-3 covers most pages; use `--topn 5` for material likely to
   be ambiguous. **Do not assert a top-1 label as certain when the top scores are
   close** - ambiguous archival pages are a known hard class, and different model
   generations legitimately disagree on them. Surface the top-N labels with scores and
   let the user (or downstream logic) decide.
3. **Input routing**: PNG/JPEG files go to single-image classification; PDFs are
   rasterized and classified page-by-page server-side. The client routes by file suffix
   automatically. Other formats (TIFF, etc.) must be converted to PNG first.
4. **Size limits**: Files over 10 MB and PDFs over 50 pages are rejected. Downscale
   large scans or split long PDFs before uploading, and tell the user you did so.
5. **Output format selection**:
   - `table` (default): human-readable summary for the conversation.
   - `csv`: for saving results or feeding downstream tabular processing.
   - `json`: for programmatic consumption preserving the raw API response.
6. **Server not reachable** (exit code 2): start it with `bash scripts/serve.sh` and
   retry. If the start script fails, inspect `api_server.log` or
   `docker compose logs` and report the cause; do not silently retry in a loop.
7. **Server errors** (exit code 3): the client already retries HTTP 502/503/504 three
   times. A persistent 5xx usually means model warmup failure (missing weights, out of
   memory) - check server logs and suggest `--version` with a single small model
   (e.g. `v2.3`) as a lower-memory fallback.
8. **Provenance**: Server-side inference is logged through the shared ATRIUM paradata
   infrastructure (`atrium_paradata.py`), so classification runs remain traceable even
   when invoked by an agent. Do not bypass the API by importing the model code directly.

## Acknowledgements & Citations 🙏

The models and dataset are developed within the [ATRIUM](https://atrium-research.eu/)
project at ÚFAL, Charles University, with data hosted on
[LINDAT/CLARIAH-CZ](https://lindat.cz). If you use this service for research, cite the
repository's `CITATION.cff` and the LINDAT dataset record
(http://hdl.handle.net/20.500.12800/1-6184).
