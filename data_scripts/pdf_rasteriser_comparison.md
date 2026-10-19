# PDF rendering for `/predict_document`: pypdfium2 against the references

**Decision (2026-09-30, [atrium-project#72](https://github.com/ufal/atrium-project/issues/72),
[#6](https://github.com/ufal/atrium-project/issues/6)):** the service renders PDF pages with
**pypdfium2** (PDFium), the whole page at `PDF_RENDER_DPI / 72` pixels per point, as RGB — the
one-call equivalent of the PyMuPDF render it replaces (`service/pdf_render.py`).

## Why not the others

| Engine                                                                 | Licence                                          | As a component of every PDF record                                                                            | Verdict                                                        |
|------------------------------------------------------------------------|--------------------------------------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| PyMuPDF (the old engine)                                               | AGPL-3.0 (or commercial)                         | would make every `/predict_document` record **AGPL-3.0** once declared; it was installed and declared nowhere | replaced                                                       |
| pdftoppm (poppler; `data_scripts/unix/pdf2png.sh`, the training pages) | GPL-2.0-or-later                                 | would make every PDF record **GPL-3.0**, and adds poppler-utils plus a subprocess per PDF to the image        | kept as the reference and for the Unix data script             |
| Ghostscript (`data_scripts/windows/pdf2png.bat`, via ImageMagick)      | AGPL-3.0                                         | as PyMuPDF                                                                                                    | host-side tool only, never in an image                         |
| **pypdfium2** (PDFium)                                                 | Apache-2.0 or BSD-3-Clause (PDFium BSD-3-Clause) | records stay **MIT** (the models' licence); declared `conditional` in `setup/para_config.txt`                 | **chosen**; atrium-alto-postprocess already reads PDFs with it |

`para_licenses` resolves a record's licence as the most restrictive of the components that ran,
and ranks GPL/AGPL above MIT, so the licence of the renderer is the licence of the result.

## What was measured

`data_scripts/compare_pdf_rasterisers.py` renders every page with the service's own code and with
the references at 300 dpi, and compares them at full size and at the classifier's input — the
page resized to 224 × 224, where a difference can change a label. PSNR in dB, higher is closer:
40 dB is a mean error of about 2.6 of 255 grey levels, 30 dB about 8. For scale, the models were
trained with colour jitter of ±50 %, Gaussian blur up to radius 2 and sharpness 0.5–1.5
(`classifier.py`), far larger than any difference below.

The first column is the one that matters: how far the service now is from **pdftoppm, the
renderer the training pages were made with**. The second is how far the old engine was.

| Set (pages)                                                                                          | pypdfium2 vs pdftoppm at 224 px: median (lowest) | PyMuPDF vs pdftoppm at 224 px | pypdfium2 vs the scan itself |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------|-------------------------------|------------------------------|
| Born-digital PDFs: alto-postprocess `CTX000000011/12.pdf`, llm-enrich `digital_born/sample.pdf` (10) | 39.6 (38.3)                                      | 39.9 (38.4)                   | —                            |
| `small_data_samples/` pages at their own size, 140–630 px wide (229)                                 | 35.9 (15.5)                                      | 32.3 (17.1)                   | 34.5 (19.2)                  |
| Slip, 74 × 105 mm, scanned at 300 dpi (10)                                                           | 74.4 (71.2)                                      | 41.5 (36.6)                   | 40.9 (36.3)                  |
| Slip, 74 × 105 mm, 600 dpi (10)                                                                      | 45.4 (42.3)                                      | 38.2 (34.1)                   | 44.8 (40.5)                  |
| Card, 95 × 61 mm, 400 dpi (10)                                                                       | 41.5 (38.7)                                      | 41.5 (38.8)                   | 50.4 (48.4)                  |
| A5, 300 dpi (10)                                                                                     | 81.3 (71.4)                                      | 47.0 (42.3)                   | 46.4 (42.0)                  |
| A4, 300 dpi (10)                                                                                     | 46.3 (43.0)                                      | 49.6 (45.0)                   | 48.1 (44.8)                  |
| A4, 400 dpi (10)                                                                                     | 52.5 (48.7)                                      | 51.7 (48.9)                   | 51.3 (48.2)                  |
| A4, 150 dpi (10)                                                                                     | 49.7 (46.9)                                      | 55.4 (50.3)                   | 47.9 (45.3)                  |
| A3 landscape, 300 dpi (10)                                                                           | 48.9 (46.1)                                      | 53.0 (49.6)                   | 51.7 (49.3)                  |
| Strip, 400 × 100 mm, 300 dpi (10)                                                                    | 84.5 (80.9)                                      | 50.0 (45.6)                   | 49.4 (45.6)                  |
| Plan, A1, 200 dpi (3)                                                                                | 59.3 (58.1)                                      | 59.8 (56.7)                   | 54.4 (53.4)                  |

The scanned sets are the sample pages placed on pages of those sizes and stored as one image per
page at the scan resolution, as a scanner writes them (`--from-images … --canvas … --scan-dpi …`).
Measured with pypdfium2 5.13.0 (PDFium 153.0.7999), PyMuPDF 1.28.2 and pdftoppm 24.02.

**Reading.** The service is as close to the training renderer as the old engine was, or closer:
closer in six sets, within 0.5 dB in three, 3–6 dB further (still 43 dB or more) in three. On the
slip, A5 and strip scans PDFium and pdftoppm produce nearly the same pixels. The low minimums of
the small pages come from 140-pixel pages whose size is not a whole number of points: every
engine resamples them, and pdftoppm is no closer to the scan there than PDFium is.

**Pixel size.** PDFium rounds a page's pixel size up from its 32-bit page size, so a page that is
2480.00005 px wide becomes 2481 px — pdftoppm and PyMuPDF round it to 2480. The model resizes
every page to its input size, so the extra column changes nothing it sees; the service checks
`MAX_IMAGE_PIXELS` on PDFium's own rounding (`service/pdf_render.pixel_size`).

## Labels on AMČR samples (2026-10-19)

The difference that decides is a changed label. It was measured on the 15 born-digital AMČR reports of
atrium-digital-convert#5 (2,514 pages, 34 to 529 per report), split into parts of 40 pages, with the service's
renderer against PyMuPDF and the ensemble of every published model (`--version all`), page-classification
`v1.9.4-beta`, on an RTX A4000 (atrium-project#72, job 7147054):

```bash
python3 data_scripts/compare_pdf_rasterisers.py <part>.pdf --labels --version all --out <part>.md
```

| Report       | Pages | Top-1 labels agree with PyMuPDF | PSNR vs PyMuPDF, full size (median) | At 224 px: median (lowest) |
|--------------|-------|---------------------------------|-------------------------------------|----------------------------|
| CTX200802472 | 282   | 280                             | 29.8                                | 46.7 (32.8)                |
| CTX200914045 | 36    | 36                              | 28.4                                | 47.0 (37.4)                |
| CTX201006925 | 85    | 85                              | 30.5                                | 48.3 (42.7)                |
| CTX201100348 | 90    | 90                              | 28.6                                | 49.8 (44.7)                |
| CTX201202683 | 178   | 178                             | 25.8                                | 47.9 (35.3)                |
| CTX201303221 | 401   | 401                             | 27.9                                | 47.7 (32.2)                |
| CTX201400303 | 431   | 429                             | 27.6                                | 43.2 (39.8)                |
| CTX201504552 | 50    | 50                              | 27.1                                | 46.3 (30.9)                |
| CTX201600915 | 106   | 106                             | 28.9                                | 45.8 (32.8)                |
| CTX201703695 | 529   | 529                             | 27.7                                | 44.7 (31.9)                |
| CTX201800066 | 75    | 75                              | 34.1                                | 52.1 (45.7)                |
| CTX201900002 | 34    | 34                              | 29.3                                | 48.8 (39.2)                |
| CTX202000362 | 83    | 83                              | 30.4                                | 50.5 (33.6)                |
| CTX202100048 | 62    | 61                              | 30.1                                | 48.4 (37.6)                |
| CTX202200566 | 72    | 72                              | 32.8                                | 49.5 (41.8)                |
| **All**      | 2,514 | **2,509 (99.8 %)**              | 28.1                                | 45.8 (30.9)                |

The five pages that differ change to a neighbouring class, never across text and image:

| Report, page     | pypdfium2 | PyMuPDF  | At 224 px |
|------------------|-----------|----------|-----------|
| CTX200802472 235 | `TEXT_P`  | `TEXT`   | 47.2      |
| CTX200802472 257 | `LINE_P`  | `DRAW`   | 34.0      |
| CTX201400303 3   | `LINE_P`  | `TEXT_P` | 45.9      |
| CTX201400303 189 | `DRAW`    | `DRAW_L` | 41.2      |
| CTX202100048 27  | `DRAW_L`  | `DRAW`   | 43.6      |

**Reading.** The renderer does not change what the service says about AMČR reports: 2,509 of 2,514 labels are
the same, and the five that move are borderline pages between two similar classes. pdftoppm, the renderer of
the training pages, was not on the GPU node, so the column against it stays empty; the scanned-set table above
already puts pypdfium2 as close to pdftoppm as PyMuPDF was. The fallback agreed on #6 (declare the
training renderer with its licence) is not needed.

To repeat it on other PDFs, in the service's environment plus PyMuPDF in a throwaway `--target` folder on
`PYTHONPATH` (and poppler-utils for the pdftoppm column):

```bash
python3 data_scripts/compare_pdf_rasterisers.py path/to/amcr_pdfs/ --labels --version all --out amcr_labels.md
```
