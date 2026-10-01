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

## Still to run: labels on AMČR samples

The difference that decides is a changed label. It needs the models (torch and the weights from
the Hugging Face Hub), which were not available where the table above was made. Run, from the
repository root, in the environment of the service plus poppler-utils (and PyMuPDF only in a
throwaway environment, if the before/after comparison is wanted):

```bash
python3 data_scripts/compare_pdf_rasterisers.py path/to/amcr_pdfs/ --labels --version all --out amcr_labels.md
```

and record the summary line of each reference here:

| Set              | Pages | Top-1 labels agree with pdftoppm | Top-1 labels agree with PyMuPDF |
|------------------|-------|----------------------------------|---------------------------------|
| AMČR sample PDFs |       |                                  |                                 |

If the labels disagree on a noticeable share of pages, the fallback agreed on #6 applies: declare
the renderer that matches the training pages, with its licence, rather than hide the difference.
