import pandas as pd
import layoutparser as lp
import cv2
import pytesseract
from gvision import GVisionAPI
from common_utils import *

pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT']  # should be preinstalled
gcv_authfile = os.environ['GCV']  # should be preset

# PNG files text extraction
class OCR_parser:
    def __init__(self, output_folder=os.environ['FOLDER_TEXTS']):
        self.ocr_output_folder = output_folder
        self.ocr_output_folder_gcv = output_folder + "_gcv"

        self.png_layout_map = {}
        self.png_file_list = []

        if not os.path.exists(self.ocr_output_folder):
            os.makedirs(self.ocr_output_folder)
        if not os.path.exists(self.ocr_output_folder_gcv):
            os.makedirs(self.ocr_output_folder_gcv)

    # get map of all image and layout files in the folders and nested folders
    def _directory_map_scraper(self, png_path: str, layout_path: str) -> dict[str, str]:
        png_list = directory_scraper(png_path, [], "png")
        layout_list = directory_scraper(layout_path, [], "txt")

        png_list.sort()
        layout_list.sort()

        for i, image in enumerate(png_list):
            self.png_layout_map[image] = layout_list[i]

        print(f"Mapped {len(self.png_layout_map.keys())} page and layout files")
        return self.png_layout_map

    # extract text from image based on the text regions in detected layout
    def _ocr_image_tesseract(self, png_file: str, layout_file: str):
        output_file = f"{self.ocr_output_folder}/{layout_file.split('/')[-1]}"

        image = cv2.imread(png_file)
        # print(image)
        # cv2.imshow("img", image)
        image = image[..., ::-1]

        try:
            layout_df = pd.read_csv(layout_file, sep="\t")
        except pd.errors.EmptyDataError:
            print(f"{layout_file} is empty")
            open(output_file, 'a').close()
            return

        layout = lp.io.load_dataframe(layout_df)

        text_blocks = lp.Layout([b for b in layout if b.type == 'TextRegion'])
        # print(text_blocks)
        for block in text_blocks:
            segment_image = (block
                             .pad(left=5, right=5, top=5, bottom=5)
                             .crop_image(image))

            text = self.model.detect(segment_image)
            # print(text)
            block.set(text=text, inplace=True)

        text_df = text_blocks.to_dataframe()

        # print(text_df)
        # print(text_df.info())

        text_df.to_csv(output_file, sep="\t", index=False)
        print(f"{len(text_df.index)} text regions were recognized and saved to {output_file}")
        return text_df

    # extract text from image using GCV
    def _ocr_image_gcv(self, png_file: str):
        img = cv2.imread(png_file)

        gvision = GVisionAPI(gcv_authfile)  # no model initialization, only API setting
        gvision.perform_request(img, request_type='text detection')

        text_df = gvision.to_df("texts")

        # print(text_df)
        # print(text_df.info())

        output_file = f"{self.ocr_output_folder}/{png_file.split('/')[-1].split('.')[0]}.txt"
        text_df.to_csv(output_file, sep="\t", index=False)
        print(f"{len(text_df.index)} text regions were recognized and saved to {output_file}")
        return text_df

    # call to process folders of images and layouts by Tesseract
    def folder_ocr_tesseract(self, png_path: str, layout_path: str) -> None:
        self.model = lp.TesseractAgent(languages='ces')  # multilingual choice available for GCV tesseract version

        png_layout_map = self._directory_map_scraper(png_path, layout_path)

        for png, layout in png_layout_map.items():
            self._ocr_image_tesseract(png, layout)

    # call to process folder of images with Google Cloud Vision
    def folder_ocr_gcv(self, png_path: str) -> None:
        png_file_list = directory_scraper(png_path, self.png_file_list)

        for img in png_file_list:
            self._ocr_image_gcv(img)
