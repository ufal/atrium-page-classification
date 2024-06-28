**Goal:** This project solves a task of Optical Character Recognition

**Scope:** Two ways of parsing PDF files to extract text were implemented, evaluation of the results and page merge are left to be done

**How to run:**

Install Google Cloud Vision and preset your auth credentials in the corresponding file

Optional: Install Tesseract

Change path to GCV credentials and/or Tesseract binaries in the [.env](.env) environment file

Change path to directories of pages, layouts, and texts in the same [.env](.env) file

Use pip to install dependencies:

``pip install -r requirements.txt``

Run [script.py](script.py) with optional flags:

``python3 script.py --pdf --img`` to run single PDF file parsing and following text extraction from its page images

``python3 script.py --pdf --img  --dir`` to parse all PDF files in the directory and extract text from all page images

``python3 script.py --img`` to extract text from a single page PNG image file, supports only GCV

``python3 script.py --img  --dir``to extract text from all page PNG image files in the directory,  supports only GCV

Add ```--nogcv``` flag to run OCR using Tesseract and LayoutParser instead of GCV

The results of PDF to PNG parsing will be saved to pages folder with page numbers added to PDF filename. Code can be found in [pdf_parser.py](pdf_parser.py) script

The results of LayputParser detection will be saved in [layouts](layouts) folder in form of tab separated tables. Code can be found in [png_layout.py](png_layout.py) script

The results of Tesseract OCR detection will be saved in [ocr_text](ocr_text) folder in form of tab separated tables. Code can be found in [ocr_parser.py](ocr_parser.py) script

The results of Google Cloud Vision OCR detection will be saved in [ocr_text_gcv](ocr_text_gcv) folder in form of tab separated tables. Code can be found in [ocr_parser.py](ocr_parser.py) script

The repository files include 2 test documents [CTX200502635.pdf](CTX200502635.pdf) and [CTX200903109.pdf](CTX200903109.pdf) referenced in [script.py](script.py)

The image preprocessing isn't finished, but examples can be found in [preprocess.py](preprocess.py)