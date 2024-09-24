import fitz
from common_utils import *


# PDF files to png pages parser
class PDF_parser:
    def __init__(self, output_folder: Path = None):
        self.layout_output_folder = Path(os.environ['FOLDER_LAYOUTS']) if output_folder is None else output_folder
        self.page_output_folder = Path(os.environ['FOLDER_PAGES'])

        self.pdf_file_list = []

        self.detector_basic = None
        self.detector_pro = None

        self.cur_file_name = ""
        self.cur_stats_summary_file = f"{self.cur_file_name}.tsv"
        self.cur_pdf_page_count = 0

        if not self.layout_output_folder.is_dir():
            self.layout_output_folder.mkdir()

        if not self.page_output_folder.is_dir():
            self.page_output_folder.mkdir()

    # load PDF, go through pages and save separate pages as PNG files
    def pdf_to_png(self, pdf_file: Path, large_size: bool = True) -> None:
        items = list(self.page_output_folder.glob(f"{self.cur_file_name}_page"))
        if len(items) == fitz.open(pdf_file).page_count:
            print(f"{pdf_file} was already split into {len(items)} page image(s)")
            return

        # high resolution image save
        zoom = 2.0 if large_size else 1.0
        mat = fitz.Matrix(zoom, zoom)

        doc = fitz.open(pdf_file)
        file_name = pdf_file.stem
        # print(doc)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            pix.save(self.page_output_folder / f"{file_name}_page_{page.number}.png")

        print(f"{file_name} was saved into {doc.page_count} page image(s)")

    # called to process directory path
    def folder_to_pages(self, folder_path: Path) -> None:
        self.pdf_file_list = directory_scraper(folder_path, "pdf")

        for file in self.pdf_file_list:
            self.pdf_to_png(Path(file))
