from pdf_parser import *
from png_layout import *
from ocr_parser import *

file_small = "CTX200903109.pdf"
file_big = "CTX200502635.pdf"

pages_output_folder = "/lnet/work/people/lutsai/pythonProject/pages"
layout_output_folder = "/lnet/work/people/lutsai/pythonProject/layouts"
ocr_output_folder = "/lnet/work/people/lutsai/pythonProject/ocr_text"
ocr_output_folder_gcv = "/lnet/work/people/lutsai/pythonProject/ocr_text_gcv"


folder_brno = "/home/lutsai/Documents/Brno-20240119T165503Z-001/Brno"
folder_arup = "/home/lutsai/Documents/ATRIUM_ARUP_vzorek/CTX"

if __name__ == "__main__":
    pdf_parser = PDF_parser(output_folder=pages_output_folder)  # turns pdf to png, TODO image preprocessing
    pdf_parser.folder_to_pages(folder_arup)  # called on folder with pdf files and folders with pdf files

    png_parser = PNG_Layout(output_folder=layout_output_folder)  # turns png to table of boxes aka layout
    png_list = png_parser.directory_scraper(pdf_parser.page_output_folder)  # called on pages_output_folder

    for image in png_list:
        png_parser.png_detect(image)  # takes time using layout parser

    ocr_parser = OCR_parser(output_folder=ocr_output_folder)  # generates raw layout tables with corresponding text prediction

    ocr_parser.folder_ocr_tesseract(pdf_parser.page_output_folder, png_parser.layout_output_folder)  # called on layouts and pages

    ocr_parser.folder_ocr_gcv(pages_output_folder)  # called on pages only