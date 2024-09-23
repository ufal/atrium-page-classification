import fitz
from common_utils import *

# PDF files to png pages parser
class PDF_parser:
    def __init__(self, output_folder=os.environ['FOLDER_PAGES']):
        self.page_output_folder = output_folder

        self.pdf_file_list = []

        if not os.path.exists(self.page_output_folder):
            os.makedirs(self.page_output_folder)

    # load PDF, go through pages and save them as PNG
    def pdf_to_png(self, pdf_file: str, large_size=True) -> None:
        items = [page_file for page_file in os.listdir(self.page_output_folder) if
                 page_file.startswith(pdf_file.split(".")[-2])]
        if len(items) == fitz.open(pdf_file).page_count:
            print(f"{pdf_file} was already split into {len(items)} page image(s)")
            return

        # high resolution image save
        zoom = 2.0 if large_size else 1.0
        mat = fitz.Matrix(zoom, zoom)

        doc = fitz.open(pdf_file)
        file_name = pdf_file.split("/")[-1].split(".")[0]
        # print(doc)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            pix.save(f"{self.page_output_folder}/{file_name}_page_{page.number}.png")

        print(f"{file_name} was saved into {doc.page_count} page image(s)")

    # called to process directory path
    def folder_to_pages(self, path: str) -> None:
        pdf_list = directory_scraper(path, self.pdf_file_list, "pdf")

        for file in pdf_list:
            self.pdf_to_png(file)
