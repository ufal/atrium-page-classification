import os
import pandas as pd
import layoutparser as lp
import cv2
import pytesseract
from gvision import GVisionAPI

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # should be preinstalled
gcv_authfile = '/home/lutsai/.config/gcloud/application_default_credentials.json'  # should be preset


class OCR_parser:
    def __init__(self, output_folder="/lnet/work/people/lutsai/pythonProject/ocr_text"):
        self.ocr_output_folder = output_folder
        self.ocr_output_folder_gcv = output_folder + "_gcv"

        self.png_layout_map = {}
        self.png_file_list = []

        if not os.path.exists(self.ocr_output_folder):
            os.makedirs(self.ocr_output_folder)
        if not os.path.exists(self.ocr_output_folder_gcv):
            os.makedirs(self.ocr_output_folder_gcv)

    def directory_map_scraper(self, png_path, layout_path):
        png_list, layout_list = [], []
        for root, dirs, files in os.walk(png_path):
            for file in files:
                if file.endswith(f".png"):
                    png_list.append(f"{root}/{file}")

        for root, dirs, files in os.walk(layout_path):
            for file in files:
                if file.endswith(f".txt"):
                    layout_list.append(f"{root}/{file}")

        png_list.sort()
        layout_list.sort()

        # print(len(png_list), len(layout_list))
        # print(png_list)
        # print(layout_list)

        for i, image in enumerate(png_list):
            self.png_layout_map[image] = layout_list[i]

        print(f"Mapped {len(self.png_layout_map.keys())} page and layout files")
        return self.png_layout_map


    def directory_scraper(self, png_path):
        for root, dirs, files in os.walk(png_path):
            for file in files:
                if file.endswith(".png"):
                    self.png_file_list.append(f"{root}/{file}")

        print(f"From directory {png_path} collected {len(self.png_file_list)} png files")
        return self.png_file_list

    def ocr_image_tesseract(self, png_file, layout_file):
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

        # print(layout_df)
        layout = lp.io.load_dataframe(layout_df)
        # print(layout)

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

    def ocr_image_gcv(self, png_file):
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

    def folder_ocr_tesseract(self, png_path, layout_path):
        self.model = lp.TesseractAgent(languages='ces')  # multilingual choice available for GCV tesseract version

        png_layout_map = self.directory_map_scraper(png_path, layout_path)

        for png, layout in png_layout_map.items():
            self.ocr_image_tesseract(png, layout)

    def folder_ocr_gcv(self, png_path):
        png_file_list = self.directory_scraper(png_path)

        for img in png_file_list:
            self.ocr_image_gcv(img)