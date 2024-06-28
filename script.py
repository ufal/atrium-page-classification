from pdf_parser import *
from png_layout import *
from ocr_parser import *
import argparse

file_small = "CTX200903109.pdf"
file_big = "CTX200502635.pdf"

pages_output_folder = os.environ['FOLDER_PAGES']
layout_output_folder = os.environ['FOLDER_LAYOUTS']
ocr_output_folder = os.environ['FOLDER_TEXTS']
ocr_output_folder_gcv = ocr_output_folder + "_gcv"

folder_brno = "/lnet/work/people/lutsai/atrium/Brno-20240119T165503Z-001/Brno"
folder_arup = "/lnet/work/people/lutsai/atrium/ATRIUM_ARUP_vzorek/CTX"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OCR PDF/PNG parser')
    parser.add_argument('-f', "--file", type=str, default=file_small, help="Single PDF file path")
    parser.add_argument('-d', "--directory", type=str, default=folder_arup, help="Path to folder with PDF files")

    parser.add_argument('-pf', "--pagefile", type=str, help="Single image file path")
    parser.add_argument('-pd', "--pagedir", type=str, default=pages_output_folder, help="Path to folder with PNG images")

    parser.add_argument("--nogcv", help="Use Tesseract instead of Google Cloud Vision", action="store_true")
    parser.add_argument("--dir", help="Process whole directory", action="store_true")
    parser.add_argument("--pdf", help="Process PDF files", action="store_true")
    parser.add_argument("--img", help="Process images", action="store_true")

    args = parser.parse_args()

    pdf_parser = PDF_parser(output_folder=pages_output_folder)  # turns pdf to png, TODO image preprocessing
    ocr_parser = OCR_parser(output_folder=ocr_output_folder_gcv)  # generates raw layout tables with corresponding text prediction

    if args.nogcv:
        png_parser = PNG_Layout(output_folder=layout_output_folder)  # turns png to table of boxes aka layout
        ocr_parser = OCR_parser(output_folder=ocr_output_folder)  # generates raw layout tables with corresponding text prediction

    if args.pdf:
        if args.dir:
            pdf_parser.folder_to_pages(args.directory)  # called on folder with pdf files and folders with pdf files
        else:
            pdf_parser.pdf_to_png(args.file)

        if args.img:
            if not args.nogcv:
                ocr_parser.folder_ocr_gcv(args.pagedir)  # called on pages only
            else:
                png_parser.folder_to_layouts(pages_output_folder)
                ocr_parser.ocr_output_folder(pages_output_folder, layout_output_folder)
    elif not args.pdf and args.img:
        if args.dir:
            ocr_parser.folder_ocr_gcv(args.pagedir)  # called on pages only
        else:
            ocr_parser._ocr_image_gcv(args.pagefile)


    # png_list = directory_scraper(pdf_parser.page_output_folder)  # called on pages_output_folder
    # for image in png_list:
    #     png_parser.png_detect(image)  # takes time using layout parser
