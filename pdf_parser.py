import os
import fitz

class PDF_parser:
    def __init__(self, output_folder="/lnet/work/people/lutsai/pythonProject/pages"):
        self.page_output_folder = output_folder

        self.pdf_file_list = []

        if not os.path.exists(self.page_output_folder):
            os.makedirs(self.page_output_folder)

    def directory_scraper(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pdf"):
                    self.pdf_file_list.append(f"{root}/{file}")

        print(f"From directory {path} collected {len(self.pdf_file_list)} pdf files")
        return self.pdf_file_list

    def pdf_to_png(self, pdf_file, large_size=True):
        items = [page_file for page_file in os.listdir(self.page_output_folder) if
                 page_file.startswith(pdf_file.split(".")[-2])]
        if len(items) == fitz.open(pdf_file).page_count:
            print(f"{pdf_file} was already split into {len(items)} page image(s)")
            return

        zoom = 2.0 if large_size else 1.0
        mat = fitz.Matrix(zoom, zoom)

        doc = fitz.open(pdf_file)
        file_name = pdf_file.split("/")[-1].split(".")[0]
        # print(doc)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            pix.save(f"{self.page_output_folder}/{file_name}_page_{page.number}.png")

        print(f"{file_name} was saved into {doc.page_count} page image(s)")
        
    def folder_to_pages(self, path):
        pdf_list = self.directory_scraper(path)

        for file in pdf_list:
            self.pdf_to_png(file)