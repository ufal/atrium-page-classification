# ATRIUM Page Classification API Service

An HTTP service that takes a page image or a PDF and returns, for each page, the best of eleven page categories with
their scores (`TEXT`, `TEXT_T`, `LINE_HW`, `DRAW`, `PHOTO`, …). When a record is sent or asked for, it also adds the
result to the page's ATRIUM document record. It serves the fine-tuned classifiers (Vision Transformer, RegNetY and
EfficientNetV2 models, trained on [historical archival pages](http://hdl.handle.net/20.500.12800/1-6184)): one
revision, or the five-model ensemble, which is the default. The service version is read from `setup/para_config.txt`
`[tool]`, its single source of truth.

The service is built with **FastAPI** and classifies single images or multipage PDFs. It sits between the PyTorch
models and downstream applications or web interfaces:

- **Several architectures.** The `version` field switches between the ViT, RegNetY and EfficientNet models.
- **GPU use.** A CUDA device is used when torch finds one; otherwise the models run on the CPU.
- **Two small frontends** for manual tests: see [Frontend(s)](#frontends).

The models and their sizes are listed in [docs/reference/models.md](../docs/reference/models.md); the eleven categories
are defined in [docs/reference/categories.md](../docs/reference/categories.md). Terms such as record, paradata and
ensemble are explained in the [ATRIUM glossary](https://ufal.github.io/atrium-project/glossary/).

On this page:
[Start](#start) ·
[Models and caches](#models-and-caches) ·
[Endpoints](#endpoints) ·
[Errors](#errors) ·
[Limits](#limits) ·
[Partial results](#partial-results) ·
[Known limits](#known-limits) ·
[Configuration (environment)](#configuration-environment) ·
[How it works](#how-it-works) ·
[Shutdown behaviour](#shutdown-behaviour) ·
[Frontend(s)](#frontends) ·
[OpenAPI (the typed contract)](#openapi-the-typed-contract) ·
[Tests](#tests)

---

## Start

**At a glance:**

| Image                                                   | CPU / GPU                                                             | Memory                                                                                                                                           | Volumes                                                                   | Calls out to                                                                                                                                      | Ready after                                                                                                                                               |
|---------------------------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ghcr.io/ufal/atrium-page-classification-api:<version>` | CPU is enough; a CUDA GPU is used when torch finds one (`--gpus all`) | about 2.44 GB of weights for the five ensemble models; `limits.memory: 4Gi` is a floor, not a comfortable fit; use in steady state: not measured | `/cache/huggingface` (the Hub cache) and `/app/model` (the saved weights) | `huggingface.co` only: the model revisions and the base-model configurations, at the first start and when an unlisted revision is first asked for | `/health` at once; `/ready` once all five models are loaded, after the first start downloads about 2.44 GB; a start with the volumes filled: not measured |

The published image first: it is what a deployment runs, and it needs no checkout.

```bash
docker run --rm -p 8000:8000 -v hf-cache:/cache/huggingface \
  ghcr.io/ufal/atrium-page-classification-api:<version>     # e.g. 1.13.0-beta; add --gpus all on a GPU node
curl -s http://localhost:8000/ready                         # 200 once the models are loaded
# from a checkout:
docker compose --profile api up --build
# or in a virtual environment (development), from the repository root:
pip install -r setup/requirements.txt -r service/requirements.txt
python -m service.api                                       # honours PORT and HOST; default 0.0.0.0:8000
uvicorn service.api:app --reload                            # hot reload, for development
```

The service answers at once and loads all five models in the background; on the first start it downloads them
first. Until they are loaded, `/health` returns 200, while `/ready` and the two `/predict_*` endpoints return 503.
Poll `/ready` until it returns 200.

From a checkout you need Python 3.11 (the image's base and the CI lane) and the two requirement files above;
NodeJS is needed only to develop the LINDAT frontend. The step-by-step setup, with the model download and a GPU image,
is in [docs/usage.md](../docs/usage.md#run-the-service-from-a-checkout). A first manual test of a running service is
under [Tests](#tests).

## Models and caches

| What                            | From where                                                                                      | Revision                                               | Into                                                                                         | Without egress                                                                                                                                                                    |
|---------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| The five fine-tuned classifiers | Hugging Face Hub, [`ufal/vit-historical-page`](https://huggingface.co/ufal/vit-historical-page) | by revision name (`v1.4` … `v5.4`, `/info` lists them) | `HF_HOME` (`/cache/huggingface` in the image); the weights are then saved under `/app/model` | fill the volume once where there is egress (`python3 run.py --hf -rev vX.4`); a model that cannot be loaded is left out of the ensemble (see [Partial results](#partial-results)) |

The models are loaded at start, while the service already answers `/health`; the image holds no weights. When
`/app/model` (the `model/` folder of a checkout) has no copy of a revision, `service/inference.py` downloads it from
the Hub by itself, and `python3 run.py --hf -rev vX.4` does the same ahead of time. Any other revision a request names
is downloaded on its first use. `GET /info` lists the five as `available_models`, with `all` for their average, and the
labels as `categories`.

## Endpoints

| Method | Path                     | Description                                                                                                                                                                                                                            |
|:-------|:-------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `GET`  | `/`                      | A short JSON welcome message pointing at `/info`. The static `index.html` interface for manual testing is served at `/frontend`.                                                                                                       |
| `GET`  | `/info`                  | Service identity + capabilities: `service`, `version`, `endpoints`, `limits` (every [limit](#limits), current value), `limits_meta` (the variable that sets each), plus available models.                                              |
| `GET`  | `/health`                | Liveness probe — 200 while the process runs, from the start and even mid-shutdown. `?deep=true` also checks at least one model version is loaded (503 on failure or while draining, with `status`, `detail`, `in_flight`, `draining`). |
| `GET`  | `/ready`                 | Readiness probe — 503 (`starting`) until the models are loaded, 200 while serving, 503 the instant `SIGTERM` arrives. The Kubernetes `readinessProbe`/`startupProbe` target.                                                           |
| `POST` | `/predict_image`         | Performs inference on an uploaded single image (any `image/*` type Pillow reads; an upload sent as `application/octet-stream` is recognised by its bytes).                                                                             |
| `POST` | `/predict_document`      | Performs inference on an uploaded multipage PDF document (`application/pdf`, or `application/octet-stream` recognised by its bytes), or the `pages` selection of it.                                                                   |
| `GET`  | `/docs`, `/openapi.json` | FastAPI's built-in interactive documentation and the live OpenAPI document.                                                                                                                                                            |

### `POST /predict_image` (multipart form)

| Field               | Default    | Notes                                                                                                                                                                                                                                                                                                                                                                                                                       |
|---------------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `file`              | *required* | The image, any `image/*` type Pillow reads (PNG, JPEG, TIFF, …). An upload sent without a media type, or as `application/octet-stream`, is taken when its bytes are an image, as the command line takes a file. Any other media type, or such an upload that is not an image, is refused with `415` `unsupported_media_type`; an `image/*` upload that cannot be read, with `422`.                                          |
| `version`           | `all`      | `all` (the ensemble) or a model revision (for example `v4.4`; `/info` `available_models` lists the published ones). A value no revision matches is refused with `422` before any model runs.                                                                                                                                                                                                                                |
| `topn`              | `3`        | Number of top predictions to return, 1–11 (one per category). Outside that range the request is refused with `422`.                                                                                                                                                                                                                                                                                                         |
| `document_json`     | none       | A baseline **ATRIUM document record** to add to (the service equivalent of the command line's `--document-json`), or an AMČR seed (`doc_id`, `source`). One that cannot be opened (not UTF-8 JSON, not an object, a newer `schema_version` major) is refused with `422` `invalid_record` before any model runs. An empty part means "no baseline" but still asks for a record, which this first-stage tool then originates. |
| `document_json_out` | `false`    | Boolean. Return a record even when no baseline is uploaded (the equivalent of `--document-json-out`). page-classification is the first stage of the pipeline, so it *originates* the record as often as it updates one.                                                                                                                                                                                                     |
| `per_model`         | `false`    | Boolean. Also return each model revision's own `topn` categories in `per_model` (per page for `/predict_document`): with `version=all`, the five votes behind the averaged `predictions`, as the command line's `--best --save-intermediates` writes them. When false, `per_model` is `null`.                                                                                                                               |

```bash
curl -X POST "http://localhost:8000/predict_image" \
  -F "file=@/path/to/image.png" \
  -F "version=v4.4" \
  -F "topn=1"
```

Example JSON response (every field is always present; `ImageResponse` in [`openapi.json`](openapi.json)):

```json
{
  "type": "image",
  "predictions": [
    {
      "label": "TEXT",
      "score": 0.975
    }
  ],
  "models_used": ["v4.4"],
  "limits_applied": [],
  "document_json": null,
  "document_json_schema_error": null,
  "paradata": {"@type": "CreateAction", "@id": "urn:uuid:…", "...": "…"}
}
```

`models_used` lists the model revisions that produced the predictions: the one asked for, or, with `version=all`,
every member of the ensemble that answered. The command line takes the same inputs the same way: `run.py -f report.pdf
--pages 1-3` renders and classifies a PDF with this service's renderer and writes the same record keys. The table of
what differs, and why, is in [docs/usage.md](../docs/usage.md#the-command-line-and-the-service).

### `POST /predict_document` (multipart form)

It takes the fields of `/predict_image`, plus:

| Field   | Default | Notes                                                                                                                                                                                                                                                                                                                                                                                                   |
|---------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `pages` | all     | Classify only these pages: 1-based positions in the PDF, for example `1,3,5-7`. Empty means every page. A malformed value, or a page past the end of the PDF, is refused with `422`. `MAX_PDF_PAGES` counts the pages classified, not the PDF's. Only these pages are in the response and in the record. digital-convert's `/describe` uses it to ask only about the pages without a usable text layer. |

It answers `{"type": "document", "pages": [{"page": 1, "predictions": [...], "models_used": [...]}, ...],
"limits_applied": [], "paradata": {...}}` (`DocumentResponse`), with `models_used` per page, and `document_json` (and
`document_json_schema_error`) only when a record was asked for. A PDF that does not open is refused with `422`. A page
the model could not classify fails the request with `500`, naming the page.

### ATRIUM Document JSON accretion

Both `POST` endpoints accept and return an optional `document_json` part. A pipeline can thus thread one per-document
record through the API exactly as it does through the command line: each stage adds its own blocks and passes the
others on.

```bash
# add to an upstream record (alto-postprocess → page-classification)
curl -X POST "http://localhost:8000/predict_image" \
  -F "file=@CTX000000001_0007.png" \
  -F "document_json=@CTX000000001.document.json" \
  -F "version=v4.4"

# or originate one (first stage, no baseline to inherit)
curl -X POST "http://localhost:8000/predict_image" \
  -F "file=@CTX000000001_0007.png" \
  -F "document_json_out=true"
```

The response then carries the updated record under `document_json`:

* only this tool's own contributions are written: the whole `page_categories` block, plus `pages[].category` and
  `pages[].category_confidence` at field level. Every other tool's block (`lines`, `entities`, `translations`, …)
  passes through byte for byte;
* the record's `doc_id` is derived as on the command line (`utils.doc_id_and_page`, which strips the page label and
  then defers to the shared `canonical_doc_id()`). A service upload and a command-line run over the same file
  therefore update the *same* record instead of forking it;
* **page keys** (`/predict_document`): a PDF's pages are classified by position, 1..N. When the baseline's page rows
  carry `page_index`, each category is written under the row with that `page_index`, and the response's page entry
  carries that label as `page_label`. Such a baseline is a born-digital record from digital-convert, whose pages are
  named by their PDF page labels (`i`, `ii`, `1`, …). A record keyed "1".."N" without `page_index` (ALTO) is keyed by
  position. A position with no row whose number is another row's label is left out of the record, never written onto
  the wrong row; its prediction is still in the response;
* a record page-classification builds that does not validate against `atrium_document.schema.json` is never
  returned: the request fails with `500` instead. A **baseline** that does not validate is still accepted (unknown
  content passes through). The response then also carries `document_json_schema_error` naming the problem, so an
  automated caller can notice without reading the service log.

## Errors

Every error has one JSON body (hub `docs/agent_skill_strategy.md` §4.4):
`{"status": <int>, "reason": <code or null>, "detail": "<text>"}`, plus `limit`
(`key`, `env`, `value`, `observed`, `unit`) when `reason` is `limit_exceeded`, and `errors`
for request-validation problems.

| Code | `reason`                 | Cause                                                                                                                                                  | What to do                                                                           |
|------|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| 413  | `limit_exceeded`         | over `MAX_UPLOAD_MB`, `MAX_PDF_PAGES` or `MAX_IMAGE_PIXELS`                                                                                            | send fewer pages (`pages`) or a smaller file, or raise the setting `limit.env` names |
| 415  | `unsupported_media_type` | the file is not an image (`/predict_image`, `accepted: ["image/*"]`) or not a PDF (`/predict_document`, `accepted: ["application/pdf"]`)               | send an image to `/predict_image`, a PDF to `/predict_document`                      |
| 422  | `invalid_record`         | the `document_json` baseline cannot be opened (not UTF-8 JSON, not an object, a newer `schema_version` major), or its `pages` is not a list of objects | send the record the previous stage wrote; update this service if the record is newer |
| 422  | `null`                   | request validation (e.g. `topn` outside 1–11), an unknown `version`, or an image or PDF that cannot be read                                            | fix the field `errors` or `detail` names; repair or re-export an unreadable file     |
| 500  | `null`                   | processing failure (a page the model could not classify included), or a record rejected by its own schema                                              | report it with `detail`; check the log for a model that failed to load               |
| 503  | `null`                   | the models are still loading, or the replica is shutting down                                                                                          | wait for `/ready` 200, or retry on another replica                                   |

```json
{"status": 413, "reason": "limit_exceeded",
 "detail": "PDF has too many pages: 73. Limit is 50 (MAX_PDF_PAGES).",
 "limit": {"key": "max_pdf_pages", "env": "MAX_PDF_PAGES", "value": 50, "observed": 73, "unit": "pages"}}
```

## Limits

Every limit this service has is an environment setting, declared once in [`tool_limits.py`](../tool_limits.py) and
reported with its current value in `GET /info` `limits`, and with the variable that sets it in `limits_meta`. A
malformed value stops the service at startup, naming the variable. Over a limit the service **refuses** the input with
`reason: "limit_exceeded"` (see [Errors](#errors)); none of this service's limits cuts an input, so every response's
`limits_applied` is `[]`. `tests/test_limits_contract.py` checks this table against `tool_limits.py` and `.env.example`.

| Key (`/info`)      | Variable           | Default   | Unit  | Over the limit                                                                            |
|--------------------|--------------------|-----------|-------|-------------------------------------------------------------------------------------------|
| `max_upload_mb`    | `MAX_UPLOAD_MB`    | 10        | MB    | 413 `limit_exceeded` — per part: the file, and the `document_json` baseline               |
| `max_pdf_pages`    | `MAX_PDF_PAGES`    | 50        | pages | 413 `limit_exceeded`, before any page is rendered — counts the pages classified (`pages`) |
| `pdf_render_dpi`   | `PDF_RENDER_DPI`   | 300       | dpi   | — (the resolution PDF pages are rendered at; 300 is how the training pages were made)     |
| `max_image_pixels` | `MAX_IMAGE_PIXELS` | 178956970 | px    | 413 `limit_exceeded`: an image, or a PDF page at `PDF_RENDER_DPI`, sized before decoding  |

Platform limits (not settings): Starlette's multipart defaults (1000 files, 1000 fields, 1 MiB
per non-file field).

## Partial results

A 200 can carry less than was asked for. Read these before using the result:

- `limits_applied` is present in every response and empty today: each limit of this service refuses
  instead of shaping the result.
- With `version=all`, a model that cannot be loaded or fails is left out, and the others are averaged.
  `models_used` (per page in `/predict_document`) lists the revisions that answered, so fewer than five
  means a reduced ensemble; the service log names the failing model, once.
- `/predict_document` classifies the pages asked for (`pages`), not the whole PDF.

## Known limits

- **The renderer differs from the training pages'.** The service renders PDF pages with pypdfium2; the
  training pages were made with pdftoppm. On 2,514 pages of 15 born-digital AMČR reports, 5 top-1
  labels differed from the ones PyMuPDF renders gave, each to a neighbouring class
  ([`data_scripts/pdf_rasteriser_comparison.md`](../data_scripts/pdf_rasteriser_comparison.md)).
- **One request classifies at most `MAX_PDF_PAGES` pages** (50 by default); a longer PDF is sent in parts
  with `pages`.
- **The models were trained on scanned archival pages.** Born-digital pages are classified, but
  their labels have not been checked against reviewed answers.

## Configuration (environment)

| Variable              | Default   | Meaning                                                                                                       |
|-----------------------|-----------|---------------------------------------------------------------------------------------------------------------|
| `PORT`                | `8000`    | port the service **binds**, and the one `service/healthcheck.py` probes                                       |
| `HOST`                | `0.0.0.0` | bind address; see the warning below                                                                           |
| `GRACEFUL_SHUTDOWN_S` | `20`      | seconds uvicorn waits for in-flight requests before closing them                                              |
| `RELOAD`              | `false`   | filesystem auto-reload — development only, never in a deployment                                              |
| `LOG_LEVEL`           | `INFO`    | root logger level for the `python -m service.api` start path                                                  |
| `ALLOWED_ORIGINS`     | `*`       | CSV of CORS origins                                                                                           |
| `MAX_UPLOAD_MB`       | `10`      | canonical upload limit — no shared default across the five services; `MAX_UPLOAD_BYTES` is read as a fallback |

Beyond the shared contract above, this service reads only its [limits](#limits), declared in
`tool_limits.py`: `service/inference.py`, `service/document_json.py` and
`service/api_client.py` contain no environment reads.
This table is the deployment-facing subset. The complete ledger — every variable this
image reads — is [`.env.example`](../.env.example) at the repo root, whose layout is
fixed by `docs/templates/env.example.template` in ufal/atrium-project. The
cross-service operator reference is `docs/k8s_deployment.md` in that same repo.

`PORT` and `HOST` are read by `service/api.py`'s `__main__` block, which is what the `api` image's `ENTRYPOINT` runs.
Setting `PORT` moves both the listener and the health probe.

`RELOAD` works under the container entrypoint (`python -m service.api`). Under a direct `python3 api.py` from
`service/` (the manual test under [Tests](#tests)) it is a no-op with a uvicorn warning. That launch
has no package context for an import string to resolve against, so the app object is passed
directly.

> **Note:** `HOST=127.0.0.1` yields a container that reports **healthy** and serves nobody:
> `service/healthcheck.py` always probes loopback by design and never reads `HOST`, so a
> loopback bind passes every probe while being unreachable from outside the container.

## How it works

Each request runs in a worker thread (`asyncio.to_thread`), not on the event loop, so a long classification never
blocks the loop. A `/predict_document` call renders and classifies its pages one after another.

**PDF rendering.** Pages are rendered with **pypdfium2** (PDFium; Apache-2.0 or BSD-3-Clause, declared in
`setup/para_config.txt`) at `PDF_RENDER_DPI` (scale `dpi / 72`), as RGB, one page at a time. What differs from the
renderer of the training pages is measured in
[`data_scripts/pdf_rasteriser_comparison.md`](../data_scripts/pdf_rasteriser_comparison.md). PDFium is not
thread-safe, so the service holds one process-wide lock while it opens and renders; the model runs outside the lock,
and concurrent requests queue only for the rendering.

**Provenance.** Every successful call returns `paradata`, the call's Process Run Crate `CreateAction`, built by
`atrium_rocrate.create_action()` as in the other ATRIUM services. `object` is the upload and, when one was sent, the
record. `result` is the record blocks the call stamped (`page_categories`, `pages`) and `predictions.json`, what it
answered with. Its `@id` is the `run_uuid` stamped into those blocks, and `/predict_document` lists `pypdfium2` among
the run's components. The service writes no paradata file.

## Shutdown behaviour

The published `api` image (`ghcr.io/ufal/atrium-page-classification-api:<version>`) declares a `HEALTHCHECK` (a
shallow `GET /health`, through the vendored `service/healthcheck.py`) and `STOPSIGNAL SIGTERM`, and sets
`GRACEFUL_SHUTDOWN_S=20`, the seconds uvicorn waits for in-flight requests.

On `SIGTERM` the service flips `GET /ready` to **503** at once, so an orchestrator stops
routing to it. It answers new `/predict_*` calls with 503 ("retry against a live replica")
and lets uvicorn finish in-flight classification before exiting. `GET /health`
deliberately stays 200 throughout: a liveness probe failing mid-shutdown would get the
container killed before the drain completed.

Inference runs in a worker thread, as [How it works](#how-it-works) says, and this matters here. uvicorn's `SIGTERM`
handler is an event-loop callback. While a synchronous `manager.predict()` held the loop, the signal could not be
processed at all, and the graceful-shutdown timeout would have nothing to measure.

A `/predict_document` call classifies up to `MAX_PDF_PAGES` (default 50) pages sequentially and
can legitimately outlive the 20 s drain budget. Raise `GRACEFUL_SHUTDOWN_S` and the
deployment's grace period together for that workload — see `docs/k8s_deployment.md` in the
hub.

A clean shutdown exits **143** (128 + SIGTERM), not 0: uvicorn re-raises the captured
signal on purpose so a supervisor sees the real cause. That is a normal stop, not a crash.

## Frontend(s)

The service includes two versions of one small web page, for immediate testing:

1. `service/frontend/`: a lightweight, standalone vanilla JS page. The service serves it at `/frontend`.
2. `service/frontend-lindat/`: the same page inside the LINDAT header and footer, a client for use inside the
   [LINDAT](https://github.com/ufal/lindat-common) ecosystem. The service does not serve it; it is tried through
   lindat-common (see [Tests](#client-side-test-with-lindat)).

## OpenAPI (the typed contract)

The service's OpenAPI document is committed as [`service/openapi.json`](openapi.json) and
attached to every release as `openapi.json` with its `openapi.json.sha256`. A client is generated from it. Every request
and response field is typed, every error response is the `ErrorBody` above, and the registered `reason` codes are
listed in `x-atrium-reason-codes`. A returned record is typed by the vendored record schema
(`AtriumDocument`). `GET /info` reports `openapi_sha256`, the digest of the spec the running
image serves — equal to the release's `openapi.json.sha256` for an image built from that tag.

- **After an API change**, regenerate and commit it — torch is not needed, the model manager is
  stubbed as the tests stub it: `python atrium_openapi.py export --app service.api:app --out
  service/openapi.json --prepare tests.openapi_contract_data:prepare`.
  `tests/test_openapi_contract.py` fails while it is stale.
- **Compatibility.** Each release compares its spec with the previous release's
  (`release.yml`, `atrium_openapi.py compare` with oasdiff). A breaking change fails the
  release unless the major version went up (for 0.x, that means 1.0), and a removed reason
  code always fails. New fields, endpoints and reason codes are additive.
- **fastapi and pydantic are pinned** exactly (`service/requirements.txt`,
  `setup/requirements-test.txt`): the spec is generated by them. Bump both by hand and regenerate.
- **The release bundle** ships `atrium_openapi.py` (the service imports it to finish and digest the
  spec) and `service/openapi.json`.

## Tests

The fast lane needs no model, no GPU and no network; install `setup/requirements-test.txt` and run
`pytest -m "not slow" --tb=short` ([CONTRIBUTING.md](../CONTRIBUTING.md#running-the-test-suite)). The service has its own
contract tests:

- `tests/test_api_contract.py` drives both endpoints with the model manager stubbed and holds every response, 200s and
  refusals, to the published schema;
- `tests/test_openapi_contract.py`, `tests/test_limits_contract.py` and `tests/test_env_contract.py` check the OpenAPI
  document, the [Limits](#limits) table and the [environment](#configuration-environment) ledger against the code;
- `tests/test_service_runtime_deps.py` asserts that every entry point the compose files and the setup script invoke is
  declared in a requirements file the image installs from.

### Try the running service

To check that the service and its models answer, open two terminal windows and run, in each:

```bash
source venv/bin/activate
cd atrium-page-classification/service/
```

Then start the API in the first window (**server console**). You should see startup logs saying the server runs on
`http://0.0.0.0:8000`:

```bash
python3 api.py
```

In the second window (**client console**) send a request. This is a top-3 classification of `image.png`, where `-f` is
the input file and `-v` the model version:

```bash
python3 api_client.py -f .../image.png -v v5.4 --top 3
```

Expected output:

```json
{
  "type": "image",
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
  ],
  "models_used": ["v5.4"],
  "limits_applied": [],
  "document_json": null,
  "document_json_schema_error": null,
  "paradata": {"@type": "CreateAction", "...": "…"}
}
```

Or, for `-v all`, the ensemble: each model's own top-`topn`, averaged, as the command line's `--best` does it:

```json
{
  "type": "image",
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
  ],
  "models_used": ["v1.4", "v2.4", "v3.4", "v4.4", "v5.4"],
  "limits_applied": [],
  "document_json": null,
  "document_json_schema_error": null,
  "paradata": {"@type": "CreateAction", "...": "…"}
}
```

### Client-side test with LINDAT

To develop the LINDAT frontend, open a **second console window** and follow these steps:

1. **Clone the repository** and place the `atrium-page-classification` project files into the `lindat-common`
   directory:

   ```bash
   git clone https://github.com/ufal/lindat-common.git
   cd lindat-common
   cp -r ../atrium-page-classification .
   ```

2. **Install the NodeJS environment** (unless you already have one) and the development dependencies:

   ```bash
   curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh | bash
   nvm install stable
   nvm use stable
   npm install
   ```

3. **Run the development server:**

   ```bash
   make run
   ```

For further details, see the **LINDAT Common Development Guide**:
[https://github.com/ufal/lindat-common/?tab=readme-ov-file#development](https://github.com/ufal/lindat-common/?tab=readme-ov-file#development).

To start the API server with hot reload (useful for development), activate the
[virtual environment](https://docs.python.org/3/library/venv.html) in your **first console window**:

```bash
cd atrium-page-classification
source venv/bin/activate
uvicorn service.api:app --reload
```

The server starts at `http://0.0.0.0:8000`; open it to use the standalone test page in `service/frontend`.

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

Open `http://localhost:8080` in your web browser to reach the LINDAT client interface. Follow the file tree to the
`atrium-page-classification/service/frontend-lindat` directory. The frontend opens and lets you upload images and test
the API.

---

Contact, licence, citation and acknowledgements:
[README](../README.md#licence-citation-and-acknowledgements).
