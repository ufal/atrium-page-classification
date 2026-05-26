# 🤝 Contributing to the Historical Document Sorting Pipeline of the ATRIUM project

Welcome! Thank you for your interest in contributing. This repository provides a robust workflow for sorting archive page images to 
facilitate further content-based processing. It addresses the common digital archive 
challenge of organizing unstructured, scanned paper sources by automatically categorizing 
them before specialized data extraction or OCR is applied.

The next step: Routing classified pages into content-specific data processing pipelines 
(e.g., machine-typed OCR, handwritten text recognition, or graphic extraction).

This document describes the project's capabilities, development workflow, code conventions, 
and rules for contributors.

## 📦 Release History

| Version     | Highlights                                                                                                                                                                                                                                      | Status      |
|:------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **v0.12.1** | Best 5 models finetuned (Averaging of results, Best models use, API draft, Data scripts)                                                                                                                                                        | Pre-release |
| **v0.11.0** | RegNetY as best - 5 models selected (Parsing of different docname-pagenum formats of image filenames, New model leaders selected based on the evaluation results, Project documentation expanded)                                               | Pre-release |
| **v0.9.0**  | ViT + EffNet + Dit+ RegNet (averaging of cross validation) (Added EfficientNetV2, RegNetY, DiT family variations, Added link to published dataset, Multiple base model families support is added, Cross-validation is included (and averaging)) | Pre-release |
| **v0.7.0**  | ViT family finetuned (Several ViT variants included, Model architecture diagram added, Fixes of previous developments)                                                                                                                          | Pre-release |
| **v0.5.0**  | ViT finetuned for page classification (Training dataset refined, Supplied project with data scripts, Switch to HF base models for finetuning to the defined dataset of labeled pages, Single ViT model released)                                | Pre-release |
| **v0.2.0**  | RFC low-dimension features (Random Forest Classifier from manually extracted image features (texture, color, etc.), Confusion matrix results are included, For the first time, labeled dataset is used, [FAILED])                               | Pre-release |
| **v0.1.0**  | DeepDoctection (OCR + DLA) draft without golden truth (PDF input file recognized for layout and OCR -> manual algorithm of page categorization, Initial sketch for the project of page classification using recognized page content, [FAILED])  | Pre-release |

## 🏗️ Project Contributions & Capabilities

This pipeline contributes 4 major stages to the data processing lifecycle, as detailed 
in the [README 📑 Table of contents](README.md#table-of-contents-).

### 1. Granular Data Management & Preparation

The project provides multiplatform utilities to transition from document-level files to 
page-level management.

* **PDF to PNG Conversion:** Automatically breaks down document-level PDF files into individual, 
page-specific PNG images using provided Unix (`.sh`) and Windows (`.bat`) scripts.
* **Automated Sorting:** Includes clustering scripts to seamlessly organize annotated images into 
category-specific subdirectories, preparing them for model training or evaluation.

### 2. Multi-Model Image Classification

Archive managers can choose from several pre-trained and fine-tuned visual models downloaded 
directly from the Hugging Face hub, allowing you to balance available hardware 
(CPU vs. GPU) with desired accuracy.

| Base Model Type              | Model Size / Name          | Best For...               | Key Feature                                                                                       |
|:-----------------------------|:---------------------------|:--------------------------|:--------------------------------------------------------------------------------------------------|
| **ViT (Vision Transformer)** | `vit-large-patch16-384`    | Maximum Accuracy          | Highest precision (99.12% Top-1) but requires more memory (1.2 GB).                               |
| **ViT (Vision Transformer)** | `vit-base-patch16-384`     | Good & Small              | Excellent balance of size (345 Mb) and accuracy (98.92% Top-1).                                   |
| **RegNetY**                  | `regnety_160.swag_ft_in1k` | Best & Small              | The recommended default (`main` branch). Fast CNN architecture with high accuracy (99.16% Top-1). |
| **EfficientNetV2**           | `efficientnetv2_m.in21k`   | Low-Resource Environments | Smallest model footprint (213 Mb) while maintaining strong performance (98.83% Top-1).            |

### 3. Archival Content Categorization

A core contribution of this project is its highly specific categorization system, 
trained on historical archaeological reports from 1920–2020. The pipeline evaluates each page 
based on three main criteria: **presence of graphical elements**, **type of text**, and 
**presence of tabular layouts**.

**Data Categories:**
* 📈 **Drawings (`DRAW`, `DRAW_L`):** Maps, schematics, and paintings (with or without tabular legends).
* 🌄 **Photos (`PHOTO`, `PHOTO_L`):** Photographs or cutouts, potentially with captions or tabular layouts.
* 📏 **Tables/Forms (`LINE_HW`, `LINE_P`, `LINE_T`):** Tabular structures containing handwritten, printed, or machine-typed text.
* 📄 **Standard Text (`TEXT_HW`, `TEXT_P`, `TEXT_T`):** Paragraphs or blocks of purely handwritten, printed, or typed text.
* 📰 **Mixed Text (`TEXT`):** Mixtures of different text types with minor graphical elements.

### 4. Inference, Ensembling, & Reporting

The project supports both single-page processing and large-scale directory batching, 
generating clean analytical data for archive managers.

* **Ensemble Learning (`averaging.py`):** A post-processing tool that combines predictions 
from different base architectures (e.g., merging a ViT run with a RegNetY run). This smooths 
out individual model errors and significantly improves accuracy on ambiguous pages without 
needing to reload heavy models.
* **Tabular Outputs:** Automatically generates detailed CSV files containing the `FILE`, 
`PAGE`, `CLASS-N` (Top-N predictions), and normalized `SCORE-N` for immediate review.
* **Visual Evaluation:** Generates confusion matrices to easily visualize inter-class errors 
and track model performance on evaluation datasets.

---

## 🌿 Branches & Environments

| Branch   | Environment          | Rule                                                                            |
|----------|----------------------|---------------------------------------------------------------------------------|
| `test`   | Staging              | Base for all development. Always branch from `test`.                            |
| `master` | Stable / Integration | Merged exclusively by a human reviewer. Do not open PRs directly into `master`. |

```text
test    ←  feature-<name>
test    ←  bugfix-<name>
master  ←  (humans only, after test stabilises)

```

### 🏷️ Branch Naming

| Type             | Pattern          | Example                  |
|------------------|------------------|--------------------------|
| New feature      | `feature-<name>` | `feature-new-model`      |
| Bug fix          | `bugfix-<name>`  | `bugfix-truncated-image` |
| Hotfix on master | `hotfix-<name>`  | `hotfix-flags-priority`  |

---

## 🔁 Contributor Workflow

1. **Create an issue** (or find an existing one) describing the problem or feature.
2. **Branch from `test`:**
```bash
git checkout test
git pull origin test
git checkout -b feature-<name>
```
3. **Implement your changes** observing the project's code conventions.
4. **Run the minimum tests** (see the Testing section).
5. **Open a Pull Request** targeting the `test` branch.

---

## 📋 Pull Request Format

Every PR must include:

* **Issue link:** `Closes #<number>` or `Refs #<number>`
* **Motivation:** why the change is needed
* **Description of change:** what was changed and how
* **Testing:** what was run, what passed, what could not be executed

Use a **Draft PR** if the work is not ready for review.

**Do not open PRs into `master` — merging into `master` is exclusively the 
maintainers' responsibility.

> **Note on issue tracking:** Issues reference the commits and PRs that resolved 
> them — not the other way around. Commit messages describe *what changed*; the issue 
> is the place to record *why* and link the resulting commits together.

---

## ✏️ Commit Messages

Format:

```text
[type] concise description of what changed
```

Allowed types:

| Type       | When to use                           |
|------------|---------------------------------------|
| `add`      | Added content (general)               |
| `edit`     | Edited existing content (general)     |
| `remove`   | Removed existing content (general)    |
| `fix`      | Bug fix                               |
| `refactor` | Refactoring without behaviour change  |
| `test`     | Adding or updating tests              |
| `docs`     | Documentation only                    |
| `chore`    | Build, dependencies, CI configuration |
| `style`    | Formatting, no logic change           |
| `perf`     | Performance optimisation              |

---

## 🧪 Code Conventions & Testing

### Code Conventions

* **Comments:** informative but short, may be LLM-generated, added when function name does 
not explain its functionality in detail
* **Argument types:** set default type (e.g., `int`, `list`) for function arguments
* **Console flags:** when a new one added, provide help message for it
* **Config files:** when set of variables changes it should be reflected in repository documentation
* **Generated code:** always should be manually launched and checked for mistakes before pushing

### Minimum checks before every commit

Always run basic validation locally before pushing:

```bash
# 1. Python compilation check
python -m compileall -q .

# 2. Pre-commit hooks (runs black, isort, flake8, etc.)
pre-commit run --all-files

```

> [!NOTE]
>  If specific scripts or extraction modules are updated, please run a smoke-test 
> against the `data_samples/` directory to verify extraction integrity.

---


### Running the test suite

The repository ships a lightweight `pytest` harness that requires **no ML models or GPU**
for standard unit tests. Heavy tests that do require models or network access are marked
`slow` and are excluded from the default run.

```bash
pip install -r requirements-test.txt  # pytest>=8.0 and pytest-cov only
```

```bash
pytest -m "not slow" --tb=short                              # fast — use before every commit
pytest --tb=short                                            # full suite (requires model setup)
pytest -m "not slow" --cov=. --cov-report=term-missing      # with coverage
```

`tests/test_paradata.py` (`ParadataLogger`, `_sanitise`) is shared across all repos.
Repo-specific modules and GPU-heavy tests are marked `@pytest.mark.slow` and skipped by default.

<details>
<summary>Test layout, per-repo targets, and fixture conventions</summary>

```text
tests/
├── __init__.py              # empty
├── conftest.py              # shared fixtures (tmp_path wrappers, sample data loaders)
├── fixtures/                # small static test-data files committed to the repo
└── test_<module>.py         # repo-specific unit tests
```

**Per-repo targets:**

| Repository                | Test file           | Primary targets                                                                                                                                    |
|---------------------------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `atrium-nlp-enrich`       | `test_keywords.py`  | `_extract_surface_text`, `_extract_lemmas`, `_extract_legacy`, `extract_keywords`, `_sort_csv_file`                                                |
| `atrium-alto-postprocess` | `test_text_util.py` | Density/ratio helpers, detectors, `pre_filter_line`, `parse_line_splits`, `categorize_line` (ppl passed directly, no GPU), `compute_quality_score` |
| `atrium-alto-postprocess` | `test_utils.py`     | `directory_scraper`, `dataframe_results` (Top-1 and Top-N), `collect_images`                                                                       |
| `atrium-translator`       | `test_utils.py`     | `_resolve_namespaces`, `validate_xml_with_xsd`, `process_alto_xml`, `process_amcr_xml` (mock translator injected)                                  |

**Slow tests** — any test loading a model checkpoint, calling an external API, or requiring a GPU must be decorated with `@pytest.mark.slow`. Document in the PR description which resource it requires and how to enable it locally.

**Fixtures** — small, self-contained files committed under `tests/fixtures/`. Tests must not read from `data_samples/` directly. Add a minimal fixture file in the same commit as any test that needs new sample data.

</details>

---

## 📁 Repository Documentation Management

Each documentation file has one target audience and one responsibility. Rules are not repeated — cross-references are used instead.

| File              | Audience        | Responsibility                                 |
|-------------------|-----------------|------------------------------------------------|
| `README.md`       | GitHub visitors | Project overview, workflow stages, quick start |
| `CONTRIBUTING.md` | Developers      | Code conventions, branches, PRs, testing       |

* **Do not duplicate rules:** if a rule is defined in `CONTRIBUTING.md`, other files 
reference it rather than copying it.
* **When changing a rule:** update the canonical source and verify that referencing files
still point correctly.

---

## 📞 Contacts & Acknowledgements

For support regarding this repository, please contact **lutsai.k@gmail.com**. 

* **Developed by:** UFAL [^1]
* **Funded & Shared by:** ATRIUM [^2] & UFAL
* **Models Hosted via:** Hugging Face [^3]

**©️ 2026 UFAL & ATRIUM**

[^1]: https://ufal.mff.cuni.cz/home-page
[^2]: https://atrium-research.eu/
[^3]: https://huggingface.co/ufal/vit-historical-page
