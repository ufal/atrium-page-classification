# 🤝 Contributing to the Historical Document Sorting Pipeline of the ATRIUM project

Welcome! This repository provides a robust workflow for sorting archive page images to 
facilitate further content-based processing. It addresses the common digital archive 
challenge of organizing unstructured, scanned paper sources by automatically categorizing 
them before specialized data extraction or OCR is applied.

The next step: Routing classified pages into content-specific data processing pipelines 
(e.g., machine-typed OCR, handwritten text recognition, or graphic extraction).

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

## 📞 Contacts & Acknowledgements

For support regarding this repository, please contact **lutsai.k@gmail.com**. 

* **Developed by:** UFAL [^1]
* **Funded & Shared by:** ATRIUM [^2] & UFAL
* **Models Hosted via:** Hugging Face [^3]

**©️ 2026 UFAL & ATRIUM**

[^1]: https://ufal.mff.cuni.cz/home-page
[^2]: https://atrium-research.eu/
[^3]: https://huggingface.co/ufal/vit-historical-page